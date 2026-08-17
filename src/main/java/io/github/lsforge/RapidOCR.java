package io.github.lsforge;

import io.github.lsforge.cal_rec_boxes.CalRecBoxes;
import io.github.lsforge.ch_ppocr_cls.TextClassifier;
import io.github.lsforge.ch_ppocr_det.TextDetector;
import io.github.lsforge.ch_ppocr_rec.TextRecognizer;
import io.github.lsforge.entity.*;
import io.github.lsforge.utils.LoadImage;
import io.github.lsforge.utils.OpencvLoader;
import io.github.lsforge.utils.ProcessImg;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class RapidOCR implements AutoCloseable {
    private final float textScore;          // 过滤阈值（调用级覆盖经 ParamConfig 传参生效，不再写回本字段）
    private final int minHeight;            // 最小高度
    private final float widthHeightRatio;   // 宽高比
    private final int maxSideLen;           // 最大边长
    private final int minSideLen;           // 最小边长

    private final boolean useDet;           // 是否使用检测
    private final boolean useCls;           // 是否使用分类
    private final boolean useRec;           // 是否使用识别

    private final boolean usePreprocessImg;     // 是否开启整图 min/max side len 预处理
    private final boolean useVerticalPadding;   // 细长图是否做垂直 letterbox 填充

    // OCR 三大模块和一个辅助工具
    private final TextDetector textDet;
    private final TextClassifier textCls;
    private final TextRecognizer textRec;
    private final CalRecBoxes calRecBoxes;
    private final LoadImage loadImage;


    public static RapidOCR create() {
        return new RapidOCR();
    }

    public static RapidOCR create(OcrConfig config) {
        return new RapidOCR(config);
    }

    public RapidOCR() {
        this(new OcrConfig());
    }

    public RapidOCR(OcrConfig config) {
        if (config.getGlobal().getOpencvLibPath() != null) {
            OpencvLoader.loadOpencvLib(config.getGlobal().getOpencvLibPath());
        } else {
            OpencvLoader.loadOpencvLib();
        }
        OcrConfig.GlobalConfig globalConfig = config.getGlobal();
        this.textScore = globalConfig.getTextScore();
        this.minHeight = globalConfig.getMinHeight();
        this.widthHeightRatio = globalConfig.getWidthHeightRatio();

        // 初始化 检测/分类/识别模块（引擎参数全局共享一份，从 Global 组装）
        this.useDet = globalConfig.isUseDet();
        this.textDet = new TextDetector(config.getDet(), OrtInferConfig.from(globalConfig, config.getDet().getModelPath()));

        this.useCls = globalConfig.isUseCls();
        this.textCls = new TextClassifier(config.getCls(), OrtInferConfig.from(globalConfig, config.getCls().getModelPath()));

        this.useRec = globalConfig.isUseRec();
        this.textRec = new TextRecognizer(config.getRec(), OrtInferConfig.from(globalConfig, config.getRec().getModelPath()));

        this.usePreprocessImg = globalConfig.isUsePreprocessImg();
        this.useVerticalPadding = globalConfig.isUseVerticalPadding();

        this.loadImage = new LoadImage();
        this.maxSideLen = globalConfig.getMaxSideLen();
        this.minSideLen = globalConfig.getMinSideLen();

        // 初始化文本框后处理模块
        this.calRecBoxes = new CalRecBoxes();
    }

    public OcrResult run(String imagePath) throws Exception {
        return this.runImpl(imagePath, new ParamConfig());
    }

    public OcrResult run(Path imagePath) throws Exception {
        return this.runImpl(imagePath, new ParamConfig());
    }

    public OcrResult run(byte[] imageData) throws Exception {
        return this.runImpl(imageData, new ParamConfig());
    }

    public OcrResult run(BufferedImage image) throws Exception {
        return this.runImpl(image, new ParamConfig());
    }

    public OcrResult run(Mat mat) throws Exception {
        return this.runImpl(mat, new ParamConfig());
    }

    public OcrResult run(String imagePath, ParamConfig paramConfig) throws Exception {
        return this.runImpl(imagePath, paramConfig);
    }

    public OcrResult run(Path imagePath, ParamConfig paramConfig) throws Exception {
        return this.runImpl(imagePath, paramConfig);
    }

    public OcrResult run(byte[] imageData, ParamConfig paramConfig) throws Exception {
        return this.runImpl(imageData, paramConfig);
    }

    public OcrResult run(BufferedImage image, ParamConfig paramConfig) throws Exception {
        return this.runImpl(image, paramConfig);
    }

    public OcrResult run(Mat mat, ParamConfig paramConfig) throws Exception {
        return this.runImpl(mat, paramConfig);
    }

    @Override
    public void close() {
        if (textDet != null) {
            textDet.close();
        }
        if (textCls != null) {
            textCls.close();
        }
        if (textRec != null) {
            textRec.close();
        }
    }

    /**
     * 执行检测 -> 分类 -> 识别 的完整流程
     *
     * @param imgContent    输入图片（支持多种格式：路径、字节数组、OpenCV Mat 等）
     * @param paramConfig   输入参数
     * @return 返回一个包含识别结果和耗时信息的自定义结构
     */
    private OcrResult runImpl(Object imgContent, ParamConfig paramConfig) throws Exception {
        long startTime = System.nanoTime();
        // 如果外部没有传值，则使用类内部的默认值
        boolean realUseDet = (paramConfig.getUseDet() == null) ? this.useDet : paramConfig.getUseDet();
        boolean realUseCls = (paramConfig.getUseCls() == null) ? this.useCls : paramConfig.getUseCls();
        boolean realUseRec = (paramConfig.getUseRec() == null) ? this.useRec : paramConfig.getUseRec();

        boolean returnWordBox = Boolean.TRUE.equals(paramConfig.getReturnWordBox());
        boolean returnWordLevel = Boolean.TRUE.equals(paramConfig.getReturnWordLevel());

        // 调用级参数：解析为局部变量随调用传递，不写回实例状态
        // （ParamConfig 语义即单次调用生效；旧实现写回实例字段，并发调用同一实例时互相覆盖）
        float realTextScore = (paramConfig.getTextScore() != null) ? paramConfig.getTextScore() : this.textScore;
        float realBoxThresh = (paramConfig.getBoxThresh() != null) ? paramConfig.getBoxThresh() : textDet.postprocessOp.boxThresh;
        float realUnclipRatio = (paramConfig.getUnclipRatio() != null) ? paramConfig.getUnclipRatio() : textDet.postprocessOp.unclipRatio;

        // 加载图片：如果用户传入 Mat，我们 clone 一份，确保内部 Mats 都由我们释放
        Mat originalImg = loadImage.call(imgContent);
        Mat img = (imgContent instanceof Mat) ? originalImg.clone() : originalImg;
        // 记录原始图像尺寸
        int rawH = originalImg.rows();
        int rawW = originalImg.cols();

        // 用于收集本方法内部创建、需要统一释放的 Mat
        List<Mat> matsToRelease = new ArrayList<>();

        // 预处理（缩放到合理大小；可通过 usePreprocessImg 关闭，关闭时比例记 1.0）
        Mat prevImg = img;
        Triple<Mat, Float, Float> processedImg = this.usePreprocessImg
                ? this.preprocess(img)
                : Triple.of(img, 1.0f, 1.0f);
        img = processedImg.getLeft();
        if (img != prevImg) {
            matsToRelease.add(prevImg);
        }
        // 预处理操作记录：缩放比 + letterbox 填充量，getOriginPoints 据此做逆变换
        PreprocessRecord rec = new PreprocessRecord();
        rec.ratioH = processedImg.getMiddle();
        rec.ratioW = processedImg.getRight();
        // 初始化输出结果
        List<Point[]> dtBoxes = null; // 检测到的文字区域
        List<Pair<String, Float>> clsRes = null; // 分类结果(方向, 置信度)
        List<TupleResult> recRes = null; // recRes: 每一张图对应 (text, confidence) 的列表
        double detElapsed = 0.0;
        double clsElapsed = 0.0;
        double recElapsed = 0.0;

        try {
            // ========== 1、检测阶段 ==========
            List<Mat> imgList = new ArrayList<>();
            if (realUseDet) {
                // 第二步：根据宽高比或最小高度条件，可能进行 letterbox 填充（可通过 useVerticalPadding 关闭）
                // 未填充时 rec.padTop/padLeft 保持 0，逆变换语义一致
                if (this.useVerticalPadding) {
                    Mat padded = this.maybeAddLetterbox(img, rec);
                    if (padded != img) {
                        matsToRelease.add(img);
                        img = padded;
                    }
                }
                // 执行文本检测
                Pair<List<Point[]>, Double> detResult = textDet.call(img, realBoxThresh, realUnclipRatio);
                dtBoxes = detResult.getLeft();       // 检测得到的文本框
                detElapsed = detResult.getRight();  // 检测所耗时间（秒）

                if (dtBoxes == null || dtBoxes.isEmpty()) {
                    // 如果没检测到任何文本框，直接返回空结果
                    double elapseSec = (System.nanoTime() - startTime) / 1e9;
                    return new OcrResult("", Collections.emptyList(), elapseSec, detElapsed, clsElapsed, recElapsed);
                }

                // 对检测结果进行排序
                dtBoxes = this.sortedBoxes(dtBoxes);

                // 裁剪出检测到的文本框区域
                imgList = this.getCropImgList(img, dtBoxes);
                matsToRelease.addAll(imgList);
            } else {
                imgList.add(img);
            }

            // ========== 2、分类阶段 ==========
            if (realUseCls) {
                Triple<List<Mat>, List<Pair<String, Float>>, Double> clsResultTriple = textCls.call(imgList);
                imgList = clsResultTriple.getLeft();  // 分类器可能帮我们转正图像
                clsRes = clsResultTriple.getMiddle();          // 分类的标签+置信度
                clsElapsed = clsResultTriple.getRight();       // 分类耗时(秒)
            }

            // ========== 3、识别阶段 ==========
            if (realUseRec) {
                // 是否返回单词级别的框
                Pair<List<TupleResult>, Double> resultBundle = textRec.call(imgList, returnWordBox);
                recRes = resultBundle.getLeft();
                recElapsed = resultBundle.getRight();
            }

            // ========== 后处理：计算 word-level boxes（可选） ==========
            if (dtBoxes != null && recRes != null && returnWordBox) {
                // 调用 calRecBoxes
                recRes = calRecBoxes.call(imgList, dtBoxes, recRes, returnWordLevel);

                // 接下来需要把坐标映射回原图 (根据预处理记录 + Padding 记录)
                for (TupleResult recResi : recRes) {
                    if (recResi.getWordBoxResult().getSortedWordBoxList() != null) {
                        List<Point[]> originPoints = this.getOriginPoints(recResi.getWordBoxResult().getSortedWordBoxList(), rec, rawH, rawW);
                        recResi.getWordBoxResult().setSortedWordBoxList(originPoints);
                    }
                }
            }

            if (dtBoxes != null) {
                dtBoxes = this.getOriginPoints(dtBoxes, rec, rawH, rawW);
            }

            double elapseSec = (System.nanoTime() - startTime) / 1e9;
            return getFinalRes(dtBoxes, clsRes, recRes, elapseSec, detElapsed, clsElapsed, recElapsed, realTextScore);
        } finally {
            // 释放内部创建的 Mat（用户传入的 Mat 不会被释放，因为已经 clone）
            for (Mat mat : matsToRelease) {
                if (mat != null) {
                    mat.release();
                }
            }
            // img 在 useDet=false 时已被加入 imgList，已在 matsToRelease 中处理；
            // useDet=true 时，img 是工作 Mat，需要释放
            if (realUseDet && img != null) {
                img.release();
            }
        }
    }

    /** 兼容旧签名：使用构造时的默认 textScore */
    public OcrResult getFinalRes(List<Point[]> dtBoxes, List<Pair<String, Float>> clsRes, List<TupleResult> recRes,
                                 double elapseSec, double detElapsed, double clsElapsed, double recElapsed) {
        return getFinalRes(dtBoxes, clsRes, recRes, elapseSec, detElapsed, clsElapsed, recElapsed, this.textScore);
    }

    private OcrResult getFinalRes(List<Point[]> dtBoxes, List<Pair<String, Float>> clsRes, List<TupleResult> recRes,
                                 double elapseSec, double detElapsed, double clsElapsed, double recElapsed, float textScore) {
        // 1. 无识别结果（recRes==null）：分类结果或检测框单独包装，strRes 为空串
        if (recRes == null) {
            List<RecResult> recResultList = new ArrayList<>();
            if (dtBoxes == null) {
                if (clsRes != null) {
                    for (Pair<String, Float> pair : clsRes) {
                        recResultList.add(new RecResult(null, pair.getLeft(), pair.getRight(), null));
                    }
                }
            } else {
                for (Point[] box : dtBoxes) {
                    recResultList.add(new RecResult(box, "", 0.0f, null));
                }
            }
            return new OcrResult("", recResultList, elapseSec, detElapsed, clsElapsed, recElapsed);
        }

        // 2. 无检测框（useDet=false）：直接包装识别结果
        if (dtBoxes == null) {
            List<RecResult> recResultList = new ArrayList<>();
            for (TupleResult tr : recRes) {
                recResultList.add(new RecResult(null, tr.getText(), tr.getConfidence(), tr.getWordBoxResult()));
            }
            return new OcrResult(joinTexts(recResultList), recResultList, elapseSec, detElapsed, clsElapsed, recElapsed);
        }

        // 3. det + rec：按置信度与文本有效性过滤后逐框组装
        Pair<List<Point[]>, List<TupleResult>> filtered = filterResult(dtBoxes, recRes, textScore);
        List<Point[]> filteredBoxes = filtered.getLeft();
        List<TupleResult> filteredRes = filtered.getRight();
        if (filteredBoxes == null || filteredRes == null || filteredBoxes.isEmpty()) {
            return new OcrResult("", Collections.emptyList(), elapseSec, detElapsed, clsElapsed, recElapsed);
        }
        List<RecResult> recResultList = new ArrayList<>();
        for (int i = 0; i < filteredBoxes.size(); i++) {
            TupleResult tr = filteredRes.get(i);
            recResultList.add(new RecResult(filteredBoxes.get(i), tr.getText(), tr.getConfidence(), tr.getWordBoxResult()));
        }
        return new OcrResult(joinTexts(recResultList), recResultList, elapseSec, detElapsed, clsElapsed, recElapsed);
    }

    /** 各行识别文本以换行拼接为汇总文本 */
    private static String joinTexts(List<RecResult> recResultList) {
        return recResultList.stream().map(RecResult::getText).collect(Collectors.joining("\n"));
    }

    /** 兼容旧签名：使用构造时的默认 textScore */
    public Pair<List<Point[]>, List<TupleResult>> filterResult(List<Point[]> dtBoxes, List<TupleResult> recRes) {
        return filterResult(dtBoxes, recRes, this.textScore);
    }

    private Pair<List<Point[]>, List<TupleResult>> filterResult(List<Point[]> dtBoxes, List<TupleResult> recRes, float textScore) {
        if (dtBoxes == null || recRes == null) {
            return Pair.of(null, null);
        }
        List<Point[]> filterBoxes = new ArrayList<>();
        List<TupleResult> filterRecRes = new ArrayList<>();

        for (int i = 0; i < dtBoxes.size() && i < recRes.size(); i++) {
            Point[] box = dtBoxes.get(i);
            TupleResult t = recRes.get(i);

            // if score >= self.text_score
            // 且识别文本非空（对齐 Python 端 valid_ids 过滤：rec 文本 strip 后为空的框一并删除）
            if (t.getConfidence() >= textScore && !isBlankText(t.getText())) {
                filterBoxes.add(box);
                filterRecRes.add(t);
            }
        }
        return Pair.of(filterBoxes, filterRecRes);
    }

    /**
     * 判断文本是否全为空白（对齐 Python str.strip() 后判空的语义：
     * Unicode 空白 + NBSP，Character.isWhitespace 不含 NBSP，需额外判断）
     */
    private static boolean isBlankText(String s) {
        if (s == null) {
            return true;
        }
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (!Character.isWhitespace(c) && c != '\u00A0') {
                return false;
            }
        }
        return true;
    }

    /**
     * 预处理：先判断图像最大边是否超过 maxSideLen，如果是则等比例缩小；
     * 再判断图像最小边是否小于 minSideLen，如果是则等比例放大。
     */
    private Triple<Mat, Float, Float> preprocess(Mat img) {
        float ratioH = 1.0f;
        float ratioW = 1.0f;

        // 最大边超出限制则缩小（失败时保持原图，比例 1.0）
        if (Math.max(img.rows(), img.cols()) > this.maxSideLen) {
            try {
                Triple<Mat, Float, Float> reduced = ProcessImg.reduceMaxSide(img, this.maxSideLen);
                img = reduced.getLeft();
                ratioH = reduced.getMiddle();
                ratioW = reduced.getRight();
            } catch (ProcessImg.ResizeImgError e) {
                e.printStackTrace();
            }
        }

        // 最小边小于限制则放大（ratio 连乘前一步的缩小比，得到总缩放比）
        if (Math.min(img.rows(), img.cols()) < this.minSideLen) {
            try {
                Triple<Mat, Float, Float> increased = ProcessImg.increaseMinSide(img, this.minSideLen);
                img = increased.getLeft();
                ratioH *= increased.getMiddle();
                ratioW *= increased.getRight();
            } catch (ProcessImg.ResizeImgError e) {
                e.printStackTrace();
            }
        }

        return Triple.of(img, ratioH, ratioW);
    }

    /**
     * 预处理操作记录：整图缩放比与 letterbox 填充量。
     * getOriginPoints 据此做逆变换（先撤销 padding、再撤销缩放），把坐标映射回原图。
     * 替代旧的 LinkedHashMap&lt;String, Object&gt; + 逆序键遍历 + 强转方案，行为等价。
     */
    private static class PreprocessRecord {
        float ratioH = 1.0f;   // 原图高 / 处理图高
        float ratioW = 1.0f;   // 原图宽 / 处理图宽
        int padTop = 0;        // letterbox 顶部填充量（像素）
        int padLeft = 0;       // letterbox 左侧填充量（像素）
    }

    /**
     * 判断是否需要添加 letterbox（当 h <= minHeight 或者 宽高比超过设定阈值时），
     * 填充量写入 rec.padTop（padLeft 恒为 0），供 getOriginPoints 逆变换。
     */
    private Mat maybeAddLetterbox(Mat img, PreprocessRecord rec) {
        int h = img.rows();
        int w = img.cols();

        boolean useLimitRatio = true;
        if (this.widthHeightRatio == -1.0f) {
            useLimitRatio = false;
        }

        if (h <= this.minHeight || (useLimitRatio && (float) w / (float) h > this.widthHeightRatio)) {
            int paddingH = getPaddingH(h, w);
            rec.padTop = paddingH;
            return ProcessImg.addRoundLetterbox(img, paddingH, paddingH, 0, 0);
        }

        // 否则不做填充（rec.padTop/padLeft 保持 0）
        return img;
    }

    /**
     * 获取需要在高度方向上填充的值
     */
    private int getPaddingH(int h, int w) {
        float targetH = Math.max((w / widthHeightRatio), (float) minHeight);
        int newH = (int) targetH * 2;
        return Math.abs(newH - h) / 2;
    }

    /** 换行判定阈值：按 y 排序后相邻框首点 y 差 >= 该值视为新的一行（对齐 Python 端 _BOX_SORT_Y_THRESHOLD） */
    static final double BOX_SORT_Y_THRESHOLD = 10;

    /**
     * 对检测结果排序：先按 y 分行（相邻 y 差 >= 10 视为换行），行内按 x 从左到右。
     * 对齐 Python 端 sorted_boxes 的行分组算法（np.lexsort((x, line_ids))），替代旧版冒泡相邻交换。
     * 每个框为 Point[4]
     */
    List<Point[]> sortedBoxes(List<Point[]> dtBoxes) {
        if (dtBoxes == null || dtBoxes.isEmpty()) {
            return dtBoxes;
        }

        // 1. 按首点 y 稳定排序（List.sort 为 TimSort，稳定）
        List<Point[]> boxes = new ArrayList<>(dtBoxes);
        boxes.sort(Comparator.comparingDouble(b -> b[0].y));

        // 2. 在 y 有序序列上分配行号：相邻 y 差 >= 阈值则行号 +1
        int n = boxes.size();
        int[] lineIds = new int[n];
        double[] xs = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i] = boxes.get(i)[0].x;
            if (i > 0) {
                double dy = boxes.get(i)[0].y - boxes.get(i - 1)[0].y;
                lineIds[i] = lineIds[i - 1] + (dy >= BOX_SORT_Y_THRESHOLD ? 1 : 0);
            }
        }

        // 3. 以 (行号, x) 为键排序，等价于 Python 的 lexsort((x, line_ids))
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> {
            int cmp = Integer.compare(lineIds[a], lineIds[b]);
            return cmp != 0 ? cmp : Double.compare(xs[a], xs[b]);
        });

        List<Point[]> result = new ArrayList<>(n);
        for (Integer idx : order) {
            result.add(boxes.get(idx));
        }
        return result;
    }

    /**
     * 根据检测框列表裁剪图像
     *
     * @param img     原始图像
     * @param dtBoxes 检测框列表，每个检测框由四个顶点组成
     * @return 裁剪后的图像列表
     */
    private List<Mat> getCropImgList(Mat img, List<Point[]> dtBoxes) {
        // 使用Stream流处理每个检测框并裁剪图像
        return dtBoxes.stream()
                .map(box -> getRotateCropImage(img, box))
                .collect(Collectors.toList());
    }

    /**
     * 根据旋转框的四个顶点裁剪图像
     *
     * @param img    原始图像
     * @param points 检测框的四个顶点
     * @return 裁剪后的图像
     */
    private Mat getRotateCropImage(Mat img, Point[] points) {
        // 计算裁剪区域的宽度
        double widthTop = distance(points[0], points[1]);
        double widthBottom = distance(points[2], points[3]);
        int imgCropWidth = (int) Math.max(widthTop, widthBottom);

        // 计算裁剪区域的高度
        double heightLeft = distance(points[0], points[3]);
        double heightRight = distance(points[1], points[2]);
        int imgCropHeight = (int) Math.max(heightLeft, heightRight);

        // 定义标准的裁剪区域四个顶点
        MatOfPoint2f ptsStd = new MatOfPoint2f(
                new Point(0, 0),
                new Point(imgCropWidth, 0),
                new Point(imgCropWidth, imgCropHeight),
                new Point(0, imgCropHeight)
        );

        // 将检测框顶点转换为MatOfPoint2f
        MatOfPoint2f ptsSrc = new MatOfPoint2f(points);

        // 计算透视变换矩阵
        Mat perspectiveTransform = Imgproc.getPerspectiveTransform(ptsSrc, ptsStd);

        // 执行透视变换裁剪图像
        Mat dstImg = new Mat();
        Imgproc.warpPerspective(img, dstImg, perspectiveTransform,
                new Size(imgCropWidth, imgCropHeight),
                Imgproc.INTER_CUBIC, Core.BORDER_REPLICATE);

        // 释放中间对象
        ptsStd.release();
        ptsSrc.release();
        perspectiveTransform.release();

        // 判断裁剪后的图像是否需要旋转（竖排文本条转横排）
        // 对齐 Python 端 np.rot90（逆时针）：实测顺时针会使竖排文本倒置，
        // 仅能靠 cls 的 180° 转正部分兜底；useCls=false 时识别全部失败
        if (((double) dstImg.rows() / dstImg.cols()) >= 1.5) {
            Mat rotated = new Mat();
            Core.rotate(dstImg, rotated, Core.ROTATE_90_COUNTERCLOCKWISE);
            dstImg.release();
            dstImg = rotated;
        }

        return dstImg;
    }

    /**
     * 计算两点之间的欧几里得距离
     *
     * @param p1 第一个点
     * @param p2 第二个点
     * @return 两点之间的距离
     */
    private double distance(Point p1, Point p2) {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y);
    }

    /**
     * 把坐标映射回原图：按预处理记录做逆变换（先撤销 letterbox 填充，再撤销整图缩放），
     * 最后裁剪到 [0, rawW] x [0, rawH]。
     *
     * @param dtBoxes 检测框列表，每个元素是一个长度为4的 Point[]
     * @param rec     预处理操作记录（缩放比 + 填充量）
     * @param rawH    原图的高度
     * @param rawW    原图的宽度
     * @return 修正后的坐标列表 (同一个对象内坐标已被就地修改)
     */
    private List<Point[]> getOriginPoints(List<Point[]> dtBoxes, PreprocessRecord rec, int rawH, int rawW) {
        // 单点逆变换：先撤销 letterbox 填充、再撤销整图缩放，最后裁剪到 [0, rawW] x [0, rawH]
        // （dtBoxes 就地修改后返回）
        for (Point[] box : dtBoxes) {
            for (Point p : box) {
                double x = (p.x - rec.padLeft) * rec.ratioW;
                double y = (p.y - rec.padTop) * rec.ratioH;
                p.x = Math.max(0, Math.min(x, rawW));
                p.y = Math.max(0, Math.min(y, rawH));
            }
        }
        return dtBoxes;
    }

}
