package io.github.lsforge.ch_ppocr_det;

import de.lighti.clipper.Clipper;
import de.lighti.clipper.ClipperOffset;
import de.lighti.clipper.Path;
import de.lighti.clipper.Paths;
import io.github.lsforge.entity.FlatTensor;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.*;

/**
 * 用于对DB模型的输出做后处理，生成文本检测框
 */
public class DBPostProcess implements AutoCloseable {

    // 二值化阈值
    private final float thresh;
    // 最低得分阈值
    public float boxThresh;
    // 最大候选框数量
    private final int maxCandidates;
    // 扩张比率
    public float unclipRatio;
    // 最小框尺寸，用于过滤小目标
    private final int minSize = 3;
    // 计算得分的模式: "fast" / "slow"
    private final String scoreMode;
    // 膨胀核
    private Mat dilationKernel = null;

    /**
     * @param thresh         二值化阈值
     * @param boxThresh      最低得分阈值
     * @param maxCandidates  最大候选框数
     * @param unclipRatio    扩张比率
     * @param scoreMode      "fast" or "slow"
     * @param useDilation    是否使用膨胀操作
     */
    public DBPostProcess(float thresh, float boxThresh, int maxCandidates, float unclipRatio, String scoreMode, boolean useDilation) {
        this.thresh = thresh;
        this.boxThresh = boxThresh;
        this.maxCandidates = maxCandidates;
        this.unclipRatio = unclipRatio;
        this.scoreMode = scoreMode != null ? scoreMode : "fast";

        if (useDilation) {
            Mat kernel = Mat.ones(new Size(2, 2), CvType.CV_8UC1);
            this.dilationKernel = kernel;
        }
    }

    @Override
    public void close() {
        if (dilationKernel != null) {
            dilationKernel.release();
            dilationKernel = null;
        }
    }

    /**
     * 对预测输出做阈值化、膨胀、找轮廓、生成检测框等
     * （boxThresh/unclipRatio 取构造时的默认值）
     */
    public ResultBundle call(FlatTensor pred, int oriHeight, int oriWidth) {
        return call(pred, oriHeight, oriWidth, this.boxThresh, this.unclipRatio);
    }

    /**
     * 对预测输出做阈值化、膨胀、找轮廓、生成检测框等（调用级阈值随参数传递，不写实例状态）
     *
     * @param pred         模型输出预测，形状 [N, 1, H, W] 的扁平张量，此处只处理 N=1 的情况
     * @param oriHeight    原始图像高度
     * @param oriWidth     原始图像宽度
     * @param boxThresh    检测框得分阈值
     * @param unclipRatio  unclip 扩张比率
     * @return 点位信息和得分
     */
    public ResultBundle call(FlatTensor pred, int oriHeight, int oriWidth, float boxThresh, float unclipRatio) {
        // pred 的形状 [N, 1, H, W]，这里只处理单张图 pred[0][0][...]
        float[] probMap = pred.getData();  // 行主序扁平，长度 H*W

        int h = pred.dim(2);       // H
        int w = pred.dim(3);       // W

        // 1. 生成二值化掩码 segmentation = pred > thresh
        byte[] maskData = new byte[h * w];
        for (int i = 0; i < maskData.length; i++) {
            float val = probMap[i];
            maskData[i] = (byte) ((val > this.thresh) ? 255 : 0);
        }
        // 一次性填充到 OpenCV 的 Mat 中，避免多次 JNI 调用
        Mat mask = new Mat(h, w, CvType.CV_8UC1);
        mask.put(0, 0, maskData);

        // 2. 若使用了膨胀操作，则在mask上执行dilate
        if (this.dilationKernel != null) {
            Imgproc.dilate(mask, mask, this.dilationKernel);
        }

        // 3. 从mask提取文本框(轮廓) + 计算score
        return this.boxesFromBitmap(probMap, h, w, mask, oriWidth, oriHeight, boxThresh, unclipRatio);
    }

    /**
     * 从二值化掩码中提取文本框 + 计算分数
     *
     * @param pred        概率图（行主序扁平，长度 h*w）
     * @param h           概率图高度
     * @param w           概率图宽度
     * @param bitmap      二值化掩码 (H, W)
     * @param destWidth   原图宽度
     * @param destHeight  原图高度
     * @param boxThresh   检测框得分阈值
     * @param unclipRatio unclip 扩张比率
     * @return 检测框列表（每框 4 点）与对应得分
     */
    private ResultBundle boxesFromBitmap(float[] pred, int h, int w, Mat bitmap, int destWidth, int destHeight, float boxThresh, float unclipRatio) {
        // 1. findContours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        // 这里相当于 Python: cv2.findContours((bitmap*255).astype(np.uint8), ...)
        // 但我们已经在上面把bitmap构建成0/255，所以直接传bitmap
        Imgproc.findContours(bitmap, contours, hierarchy,
                Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        // 限制最大候选框数量
        int numContours = Math.min(contours.size(), this.maxCandidates);

        List<Point[]> boxes = new ArrayList<>();
        List<Float> scores = new ArrayList<>();

        // 2. 遍历轮廓
        for (int i = 0; i < numContours; i++) {
            MatOfPoint2f contour2f = new MatOfPoint2f(contours.get(i).toArray());
            // 把四个点按 x 坐标排序并处理 y 坐标顺序，得到 box
            BoxAndSize boxAndSize = getMiniBoxes(contour2f);
            contour2f.release();
            Point[] box = boxAndSize.box;
            float sside = boxAndSize.minSideLen;
            if (sside < this.minSize) {
                continue;
            }

            // 2.3 计算score
            float score;
            if ("fast".equalsIgnoreCase(this.scoreMode)) {
                score = boxScoreFast(pred, h, w, box);
            } else {
                score = boxScoreSlow(pred, h, w, contours.get(i).toArray());
            }
            if (score < boxThresh) {
                continue;
            }

            // 2.4 unclip扩展
            Point[] expanded = unclip(box, unclipRatio);
            if (expanded.length == 0) {
                continue;
            }
            // 再获取新的最小外接矩形
            MatOfPoint2f expanded2f = new MatOfPoint2f(expanded);
            BoxAndSize boxAndSize2 = getMiniBoxes(expanded2f);
            expanded2f.release();
            Point[] newBox = boxAndSize2.box;
            float sside2 = boxAndSize2.minSideLen;
            if (sside2 < this.minSize + 2) {
                continue;
            }

            // 2.5 映射回原图
            Point[] mapped = mapToOriginalSize(newBox, w, h, destWidth, destHeight);
            boxes.add(mapped);
            scores.add(score);
        }

        // 释放轮廓与层级矩阵
        for (MatOfPoint contour : contours) {
            contour.release();
        }
        hierarchy.release();
        bitmap.release();

        return new ResultBundle(boxes, scores);
    }

    /**
     * 将四个点 newBox 映射回原图大小
     */
    private Point[] mapToOriginalSize(Point[] newBox, int curW, int curH, int oriW, int oriH) {
        // newBox[i].x, newBox[i].y 在 [0, curW], [0, curH] 范围
        // 要映射到 [0, oriW], [0, oriH]
        Point[] mapped = new Point[4];
        for (int i = 0; i < 4; i++) {
            double x = Math.round((newBox[i].x / (double) curW) * oriW);
            double y = Math.round((newBox[i].y / (double) curH) * oriH);
            // clip到范围
            double clippedX = Math.max(0, Math.min(x, oriW));
            double clippedY = Math.max(0, Math.min(y, oriH));
            mapped[i] = new Point(clippedX, clippedY);
        }
        return mapped;
    }

    /**
     * 取多边形的最小外接矩形，四顶点按左上、右上、右下、左下顺序返回。
     */
    public static BoxAndSize getMiniBoxes(MatOfPoint2f point2f) {
        MatOfPoint2f contour = new MatOfPoint2f(point2f);
        RotatedRect boundingBox = Imgproc.minAreaRect(contour);
        contour.release();

        Point[] points = new Point[4];
        boundingBox.points(points);
        Arrays.sort(points, Comparator.comparingDouble(p -> p.x));

        // 确定四个顶点的顺序
        int index1, index2, index3, index4;
        if (points[1].y > points[0].y) {
            index1 = 0;
            index4 = 1;
        } else {
            index1 = 1;
            index4 = 0;
        }

        if (points[3].y > points[2].y) {
            index2 = 2;
            index3 = 3;
        } else {
            index2 = 3;
            index3 = 2;
        }

        Point[] box = new Point[]{
                points[index1], points[index2], points[index3], points[index4]
        };

        float minSide = (float) Math.min(boundingBox.size.width, boundingBox.size.height);  // 计算最小边长
        return new BoxAndSize(box, minSide);
    }

    private double dist(Point a, Point b) {
        return Math.hypot(a.x - b.x, a.y - b.y);
    }

    /**
     * 计算文本框平均得分(FAST)，对应 Python box_score_fast
     *
     * @param bitmap 概率图（行主序扁平，长度 h*w）
     * @param h      概率图高度
     * @param w      概率图宽度
     * @param box    4个点
     * @return 平均得分
     */
    private float boxScoreFast(float[] bitmap, int h, int w, Point[] box) {
        double xmin = Double.MAX_VALUE, xmax = -Double.MAX_VALUE;
        double ymin = Double.MAX_VALUE, ymax = -Double.MAX_VALUE;
        for (Point p : box) {
            if (p.x < xmin) {
                xmin = p.x;
            }
            if (p.x > xmax) {
                xmax = p.x;
            }
            if (p.y < ymin) {
                ymin = p.y;
            }
            if (p.y > ymax) {
                ymax = p.y;
            }
        }

        // clip
        int xMin = (int) Math.max(0, Math.min(Math.floor(xmin), w - 1));
        int xMax = (int) Math.max(0, Math.min(Math.ceil(xmax), w - 1));
        int yMin = (int) Math.max(0, Math.min(Math.floor(ymin), h - 1));
        int yMax = (int) Math.max(0, Math.min(Math.ceil(ymax), h - 1));

        if (xMax <= xMin || yMax <= yMin) {
            return 0.0f;
        }

        // 构造一个Mask
        Mat mask = Mat.zeros(yMax - yMin + 1, xMax - xMin + 1, CvType.CV_8UC1);

        // 将 box 的坐标平移
        Point[] shifted = new Point[4];
        for (int i = 0; i < 4; i++) {
            shifted[i] = new Point(box[i].x - xMin, box[i].y - yMin);
        }

        // fillPoly => 1
        MatOfPoint mop = new MatOfPoint();
        mop.fromArray(shifted);
        List<MatOfPoint> listOfContours = Collections.singletonList(mop);
        Imgproc.fillPoly(mask, listOfContours, new Scalar(1));

        float avg = maskMean(bitmap, w, mask, xMin, yMin);
        mask.release();
        mop.release();
        return avg;
    }

    /**
     * 计算文本框平均得分(SLOW)，基于多边形本体
     */
    private float boxScoreSlow(float[] bitmap, int h, int w, Point[] pts) {
        // 找xmin, xmax, ymin, ymax
        double xmin = Double.MAX_VALUE, xmax = -Double.MAX_VALUE;
        double ymin = Double.MAX_VALUE, ymax = -Double.MAX_VALUE;
        for (Point p : pts) {
            if (p.x < xmin) {
                xmin = p.x;
            }
            if (p.x > xmax) {
                xmax = p.x;
            }
            if (p.y < ymin) {
                ymin = p.y;
            }
            if (p.y > ymax) {
                ymax = p.y;
            }
        }

        int xMin = (int) Math.max(0, Math.min(Math.floor(xmin), w - 1));
        int xMax = (int) Math.max(0, Math.min(Math.ceil(xmax), w - 1));
        int yMin = (int) Math.max(0, Math.min(Math.floor(ymin), h - 1));
        int yMax = (int) Math.max(0, Math.min(Math.ceil(ymax), h - 1));

        if (xMax <= xMin || yMax <= yMin) {
            return 0.0f;
        }

        // 构造一个Mask
        Mat mask = Mat.zeros(yMax - yMin + 1, xMax - xMin + 1, CvType.CV_8UC1);

        // 将轮廓坐标平移
        Point[] shifted = new Point[pts.length];
        for (int i = 0; i < pts.length; i++) {
            shifted[i] = new Point(pts[i].x - xMin, pts[i].y - yMin);
        }
        MatOfPoint mop = new MatOfPoint(shifted);
        List<MatOfPoint> conts = Collections.singletonList(mop);
        Imgproc.fillPoly(mask, conts, new Scalar(1));

        float avg = maskMean(bitmap, w, mask, xMin, yMin);
        mask.release();
        mop.release();
        return avg;
    }

    /**
     * 计算 bitmap 在 mask==1 区域内的平均值。
     * mask 一次性批量读出 byte[]（一次 JNI），替代逐像素 mask.get(row, col)；
     * 累加次序保持行主序，与旧实现位一致。
     *
     * @param bitmap 概率图（行主序扁平）
     * @param w      概率图宽度（用于行基址计算）
     * @param mask   二值掩码（0/1）
     * @param xMin   mask 对应的全局起始列
     * @param yMin   mask 对应的全局起始行
     */
    private float maskMean(float[] bitmap, int w, Mat mask, int xMin, int yMin) {
        int mh = mask.rows();
        int mw = mask.cols();
        byte[] maskData = new byte[mh * mw];
        mask.get(0, 0, maskData);

        double sumVal = 0.0;
        int count = 0;
        int idx = 0;
        for (int row = 0; row < mh; row++) {
            int base = (yMin + row) * w + xMin;
            for (int col = 0; col < mw; col++) {
                if (maskData[idx++] == 1) {
                    sumVal += bitmap[base + col];
                    count++;
                }
            }
        }
        if (count == 0) {
            return 0.0f;
        }
        return (float) (sumVal / count);
    }

    /**
     * 使用Clipper进行 unclip 扩展，仿照 Python 中 pyclipper + shapely
     *
     * @param box          原始四边形 (4个点)
     * @param unclipRatio  扩张比率
     * @return 扩展后的新多边形点集
     */
    private Point[] unclip(Point[] box, double unclipRatio) {
        // 1. area/perimeter 计算
        double area = polygonArea(box);
        double perimeter = polygonPerimeter(box);
        double distance = area * unclipRatio / perimeter;

        // 转换为Clipper库需要的路径格式
        Path clipperPath = new Path();
        for (Point p : box) {
            clipperPath.add(new de.lighti.clipper.Point.LongPoint((long) (p.x), (long) (p.y)));
        }
        // 使用Clipper库进行扩展
        ClipperOffset offset = new ClipperOffset();
        offset.addPath(clipperPath, Clipper.JoinType.ROUND, Clipper.EndType.CLOSED_POLYGON);
        Paths expandedPaths = new Paths();
        offset.execute(expandedPaths, distance);

        if (expandedPaths.isEmpty()) {
            return new Point[0];
        }
        // 取第一个扩展后的路径
        List<de.lighti.clipper.Point.LongPoint> expandedPath = expandedPaths.get(0);
        Point[] result = expandedPath.stream()
                .map(p -> new Point(p.getX(), p.getY()))
                .toArray(Point[]::new);
        return result;
    }


    /**
     * 计算多边形的面积
     */
    private double polygonArea(Point[] box) {
        double area = 0.0;
        for (int i = 0; i < box.length; i++) {
            int j = (i + 1) % box.length;
            area += box[i].x * box[j].y - box[j].x * box[i].y;
        }
        return Math.abs(area / 2.0);
    }

    /**
     * 计算多边形的周长
     */
    private double polygonPerimeter(Point[] box) {
        double peri = 0.0;
        for (int i = 0; i < box.length; i++) {
            int j = (i + 1) % box.length;
            peri += dist(box[i], box[j]);
        }
        return peri;
    }

    /**
     * 用于存储 (box, minSideLen)
     */
    private static class BoxAndSize {
        Point[] box;
        float minSideLen;
        public BoxAndSize(Point[] box, float minSideLen) {
            this.box = box;
            this.minSideLen = minSideLen;
        }
    }

    /**
     * 返回结果的简单封装
     * boxes: 每个框四个点的 [ [x0,y0], [x1,y1], [x2,y2], [x3,y3] ]
     * scores: 每个框的得分
     */
    public static class ResultBundle {
        private final List<Point[]> boxes;
        private final List<Float> scores;

        public ResultBundle(List<Point[]> boxes, List<Float> scores) {
            this.boxes = boxes;
            this.scores = scores;
        }

        public List<Point[]> getBoxes() {
            return boxes;
        }

        public List<Float> getScores() {
            return scores;
        }
    }

}
