package io.github.lsforge.ch_ppocr_det;

import ai.onnxruntime.OrtException;
import io.github.lsforge.entity.FlatTensor;
import io.github.lsforge.entity.OcrConfig;
import io.github.lsforge.entity.Pair;
import io.github.lsforge.entity.OrtInferConfig;
import io.github.lsforge.utils.OrtInferSession;
import org.opencv.core.Mat;
import org.opencv.core.Point;

import java.util.*;

/**
 * 文本检测
 */
public class TextDetector implements AutoCloseable {

    private final String limitType;        // "min" / "max" 等限制类型
    private final int limitSideLen;        // 限制边长
    private DetPreProcess preprocessOp;    // 预处理对象
    public final DBPostProcess postprocessOp;  // 后处理对象
    private final OrtInferSession infer;         // ONNX 推理会话

    /**
     * 初始化预处理、后处理和推理模块。
     *
     * @param detConfig      配置项
     * @param ortInferConfig 推理会话配置（引擎参数来自 Global，由门面组装）
     */
    public TextDetector(OcrConfig.DetConfig detConfig, OrtInferConfig ortInferConfig) {
        // 1. 获取 limitType / limitSideLen
        this.limitType = detConfig.getLimitType();
        this.limitSideLen = detConfig.getLimitSideLen();

        // 2. 后处理参数
        float thresh = detConfig.getThresh();
        float boxThresh = detConfig.getBoxThresh();
        int maxCandidates = detConfig.getMaxCandidates();
        float unclipRatio =detConfig.getUnclipRatio();
        boolean useDilation = detConfig.useDilation;
        String scoreMode = detConfig.getScoreMode();

        // 初始化后处理
        this.postprocessOp = new DBPostProcess(thresh, boxThresh, maxCandidates, unclipRatio, scoreMode, useDilation);

        // 初始化推理会话
        this.infer = new OrtInferSession(ortInferConfig);
    }

    /**
     * 对输入图像进行文本检测（boxThresh/unclipRatio 取后处理构造时的默认值）
     */
    public Pair<List<Point[]>, Double> call(Mat img) throws OrtException {
        return call(img, this.postprocessOp.boxThresh, this.postprocessOp.unclipRatio);
    }

    /**
     * 对输入图像进行文本检测（调用级阈值，随参数传递不写实例状态，并发安全）
     *
     * @param img          输入图像 (OpenCV Mat)
     * @param boxThresh    检测框得分阈值（ParamConfig.boxThresh 解析值）
     * @param unclipRatio  unclip 扩张比率（ParamConfig.unclipRatio 解析值）
     * @return (检测到的文本框, 处理时间)，其中文本框可为 null 若无结果
     * @throws OrtException 异常
     */
    public Pair<List<Point[]>, Double> call(Mat img, float boxThresh, float unclipRatio) throws OrtException {
        long startTime = System.nanoTime();

        if (img == null || img.empty()) {
            throw new IllegalArgumentException("img is null or empty");
        }

        // 原图的尺寸
        int oriHeight = img.rows();
        int oriWidth = img.cols();

        // 初始化预处理操作
        this.preprocessOp = getPreprocess(Math.max(oriHeight, oriWidth));

        // 执行预处理 => 返回 [1, C, H, W] 扁平张量
        FlatTensor preproImg = this.preprocessOp.call(img);
        if (preproImg == null) {
            return Pair.of(null, 0.0);
        }

        // 执行 ONNX 推理
        // preds 形状假设是 [1, 1, H, W]
        FlatTensor preds = infer.run(preproImg.getData(), preproImg.getShape());
        // 把 preds 传给后处理
        DBPostProcess.ResultBundle resultBundle = postprocessOp.call(preds, oriHeight, oriWidth, boxThresh, unclipRatio);

        // resultBundle 里包含 (boxes, scores)
        List<Point[]> dtBoxes = resultBundle.getBoxes();

        // 过滤文本框
        List<Point[]> filtered = filterTagDetRes(dtBoxes, new int[]{oriHeight, oriWidth});

        double elapseSec = (System.nanoTime() - startTime) / 1e9;
        return Pair.of(filtered, elapseSec);
    }

    /**
     * 根据图像的最大边长计算预处理对象
     * 模拟 Python  get_preprocess
     *
     * @param maxWh 图像的最大边
     * @return DetPreProcess
     */
    private DetPreProcess getPreprocess(int maxWh) {
        int actualLimitSideLen;
        if ("min".equalsIgnoreCase(this.limitType)) {
            actualLimitSideLen = this.limitSideLen;
        } else {
            // if max_wh < 960 => 960; elif <1500 =>1500; else=>2000
            if (maxWh < 960) {
                actualLimitSideLen = 960;
            } else if (maxWh < 1500) {
                actualLimitSideLen = 1500;
            } else {
                actualLimitSideLen = 2000;
            }
        }
        return new DetPreProcess(actualLimitSideLen, this.limitType);
    }

    /**
     * 过滤掉过小的文本框、并保证点坐标在图像内
     *
     * @param dtBoxes    检测到的文本框 (每个框 4个点)
     * @param imageShape [height, width]
     * @return 过滤后的文本框列表
     */
    private List<Point[]> filterTagDetRes(List<Point[]> dtBoxes, int[] imageShape) {
        int imgHeight = imageShape[0];
        int imgWidth = imageShape[1];

        List<Point[]> filtered = new ArrayList<>();
        for (Point[] box : dtBoxes) {
            // 1. 对四点顺时针排序
            Point[] ordered = orderPointsClockwise(box);

            // 2. clip到图像范围
            clipDetRes(ordered, imgHeight, imgWidth);

            // 3. 计算边长
            double w = dist(ordered[0], ordered[1]);
            double h = dist(ordered[0], ordered[3]);

            if (w <= 3 || h <= 3) {
                continue;
            }
            filtered.add(ordered);
        }

        return filtered;
    }

    /**
     * 对4个点按顺时针排序
     *
     * @param pts 4个点 [ Point(x0,y0), Point(x1,y1), Point(x2,y2), Point(x3,y3) ]
     * @return 排序后的4点
     */
    private Point[] orderPointsClockwise(Point[] pts) {
        if (pts == null || pts.length != 4) {
            throw new IllegalArgumentException("输入的点数组必须包含4个点");
        }

        // 创建一个副本以避免修改原始数组
        Point[] sortedPts = pts.clone();

        // 1. 先按 x 坐标排序
        Arrays.sort(sortedPts, Comparator.comparingDouble(p -> p.x));
        // 前2是左Most，后2是右Most
        Point[] leftMost = new Point[]{sortedPts[0], sortedPts[1]};
        Point[] rightMost = new Point[]{sortedPts[2], sortedPts[3]};

        // 2. 对 leftMost 按 y 排序，得到 (tl, bl)
        Arrays.sort(leftMost, Comparator.comparingDouble(p -> p.y));
        Point tl = leftMost[0];
        Point bl = leftMost[1];

        // 3. 对 rightMost 按 y 排序，得到 (tr, br)
        Arrays.sort(rightMost, Comparator.comparingDouble(p -> p.y));
        Point tr = rightMost[0];
        Point br = rightMost[1];

        // 顺时针: [tl, tr, br, bl]
        return new Point[]{tl, tr, br, bl};
    }

    /**
     * 将文本框的点坐标 clip 到图像范围内
     *
     * @param points 4个点
     * @param imgH   图像高
     * @param imgW   图像宽
     */
    private void clipDetRes(Point[] points, int imgH, int imgW) {
        for (Point point : points) {
            // Clip x 坐标到 [0, imgW - 1]
            point.x = Math.max(0, Math.min(point.x, imgW - 1));
            // Clip y 坐标到 [0, imgH - 1]
            point.y = Math.max(0, Math.min(point.y, imgH - 1));
        }
    }

    @Override
    public void close() {
        if (infer != null) {
            infer.close();
        }
        if (postprocessOp != null) {
            postprocessOp.close();
        }
    }

    /**
     * 计算两点间的距离
     */
    private double dist(Point p1, Point p2) {
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        return Math.hypot(dx, dy);
    }

}
