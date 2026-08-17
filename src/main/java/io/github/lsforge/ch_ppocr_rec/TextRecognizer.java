package io.github.lsforge.ch_ppocr_rec;

import ai.onnxruntime.OrtException;
import io.github.lsforge.entity.OcrConfig;
import io.github.lsforge.entity.OrtInferConfig;
import io.github.lsforge.entity.Pair;
import io.github.lsforge.entity.TupleResult;
import io.github.lsforge.utils.OrtInferSession;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.*;
import java.util.stream.IntStream;

public class TextRecognizer implements AutoCloseable {

    private final OrtInferSession session;
    private final CTCLabelDecode postprocessOp;

    private final int recBatchNum;         // 识别批处理大小
    private final int[] recImageShape;     // 识别输入形状 (如 [3, 32, 320])

    /**
     * @param recConfig      识别模块配置
     * @param ortInferConfig 推理会话配置（引擎参数来自 Global，由门面组装）
     */
    public TextRecognizer(OcrConfig.RecConfig recConfig, OrtInferConfig ortInferConfig) {
        // 1. 创建 ONNX 推理会话
        this.session = new OrtInferSession(ortInferConfig);

        // 2. 判断是否可从会话内获取字符表
        List<String> character = null;
        if (this.session.haveKey("character")) {
            character = this.session.getCharacterList("character");
        }

        // 3. 获取自定义字符文件路径
        String characterPath = recConfig.getRecKeysPath();

        // 4. 初始化 CTC 后处理类
        this.postprocessOp = new CTCLabelDecode(character, characterPath);

        // 5. 批处理数量 & 图像形状
        this.recBatchNum = recConfig.getRecBatchNum();
        // 例如 [3, 32, 320]
        this.recImageShape = recConfig.getRecImgShape();
    }

    @Override
    public void close() {
        if (session != null) {
            session.close();
        }
    }

    /**
     * 进行文本识别
     *
     * @param imgList       多张图像的列表
     * @param returnWordBox 是否返回单词边界信息
     * @return (识别结果列表, 总耗时)
     */
    public Pair<List<TupleResult>, Double> call(List<Mat> imgList, boolean returnWordBox) throws OrtException {
        // 若只有单张图，可封装成 List
        if (imgList == null || imgList.isEmpty()) {
            return Pair.of(Collections.emptyList(), 0.0);
        }

        // 1. 计算每张图像的宽高比
        double[] widthArray = new double[imgList.size()];
        for (int i = 0; i < imgList.size(); i++) {
            Mat img = imgList.get(i);
            double w = img.width() * 1.0;
            double h = img.height() * 1.0;
            widthArray[i] = w / h;
        }

        // 2. 对宽高比排序，返回索引
        Integer[] indices = IntStream.range(0, widthArray.length)
                .boxed()
                .toArray(Integer[]::new);
        Arrays.sort(indices, Comparator.comparingDouble(o -> widthArray[o]));

        // 3. 准备结果数组，用来存放每张图像的识别结果
        TupleResult[] recRes = new TupleResult[imgList.size()];
        // 先初始化空值
        for (int i = 0; i < recRes.length; i++) {
            recRes[i] = new TupleResult("", 0.0f, null);
        }

        // 4. 批量处理
        int imgNum = imgList.size();
        double totalElapse = 0.0;

        for (int beg = 0; beg < imgNum; beg += recBatchNum) {
            int end = Math.min(imgNum, beg + recBatchNum);

            // 获取网络输入形状(如 3, 32, 320)
            int imgC = recImageShape[0];
            int imgH = recImageShape[1];
            int imgW = recImageShape[2];

            // 在批次内部，统计最大宽高比
            float maxWhRatio = (float) imgW / imgH;
            List<Float> whRatioList = new ArrayList<>();

            for (int i = beg; i < end; i++) {
                int idx = indices[i];
                Mat curImg = imgList.get(idx);
                float curH = curImg.height();
                float curW = curImg.width();
                float whRatio = curW / curH;
                if (whRatio > maxWhRatio) {
                    maxWhRatio = whRatio;
                }
                whRatioList.add(whRatio);
            }

            // 5. 归一化 & 调整图像大小，直接写入批次的扁平数组 [n, C, H, W]
            //    注意：批次宽度为动态值（随 maxWhRatio 增大，下限为配置的 imgW/imgH），
            //    与 resizeNormImg 内部的 imgWidth 计算保持同一表达式
            int batchSize = end - beg;
            int batchW = (int) (imgH * maxWhRatio);
            int sampleSize = imgC * imgH * batchW;
            float[] normImgBatch = new float[batchSize * sampleSize];
            for (int i = beg; i < end; i++) {
                int idx = indices[i];
                Mat curImg = imgList.get(idx);
                resizeNormImg(curImg, maxWhRatio, normImgBatch, (i - beg) * sampleSize);
            }

            // 6. 推理（输出保持嵌套数组形态——rec 输出量大，拼平的 arraycopy 过路费不划算）
            long startTime = System.currentTimeMillis();
            float[][][] preds = (float[][][]) session.runBoxed(normImgBatch, new long[]{batchSize, imgC, imgH, batchW});
            double elapseSec = (System.currentTimeMillis() - startTime) / 1000.0;
            totalElapse += elapseSec;

            // 7. 后处理
            List<DecodeResult> recResult = postprocessOp.call(preds, returnWordBox, whRatioList, maxWhRatio);

            // 8. 将结果放回到对应的下标
            for (int rno = 0; rno < recResult.size(); rno++) {
                DecodeResult dr = recResult.get(rno);
                int origIdx = indices[beg + rno];
                recRes[origIdx] = new TupleResult(dr.getText(), dr.getConfidence(), dr.getWordBoxInfo());
            }
        }

        // 返回 (识别结果, 总耗时)
        List<TupleResult> resultList = Arrays.asList(recRes);
        return Pair.of(resultList, totalElapse);
    }

    /**
     * 调整和归一化图像，结果以 CHW 行主序写入 out 的 offset 偏移处。
     * resize/类型转换在 OpenCV 内完成，随后一次批量读取（单次 JNI），
     * 归一化 (v/255)*2-1 与 HWC->CHW 交织在 Java 内进行（保持原浮点表达式，数值位不变）。
     *
     * @param imgMat     OpenCV Mat 格式的图像
     * @param maxWhRatio 本批次最大宽高比
     * @param out        批次扁平数据（[n, C, H, W] 中本样本的 CHW 段）
     * @param offset     本样本在 out 中的起始偏移
     */
    public void resizeNormImg(Mat imgMat, float maxWhRatio, float[] out, int offset) {
        int imgChannel = recImageShape[0];
        int imgHeight = recImageShape[1];
        // int imgWidth = recImageShape[2];

        // 确保输入图像的通道数与预期一致
        if (imgMat.channels() != imgChannel) {
            throw new IllegalArgumentException("输入图像的通道数与预期不符");
        }

        // 计算新的图像宽度
        int imgWidth = (int) (imgHeight * maxWhRatio);

        // 获取原图的高度和宽度
        int originalHeight = imgMat.rows();
        int originalWidth = imgMat.cols();

        // 计算宽高比
        float ratio = (float) originalWidth / (float) originalHeight;

        // 判断调整后的宽度是否超过限制
        int resizedW;
        if (Math.ceil(imgHeight * ratio) > imgWidth) {
            resizedW = imgWidth;
        } else {
            resizedW = (int) Math.ceil(imgHeight * ratio);
        }

        // 调整图像大小
        Size size = new Size(resizedW, imgHeight);
        Mat resizedMat = new Mat();
        Imgproc.resize(imgMat, resizedMat, size);

        // 将图像转换为浮点类型（不缩放，归一化在下方按原表达式计算）
        Mat resizedFloatMat = new Mat();
        resizedMat.convertTo(resizedFloatMat, CvType.CV_32FC(imgChannel));

        // 一次性批量读取整幅 HWC 数据（右侧 padding 区域保持 out 预置的 0）
        float[] hwc = new float[imgChannel * imgHeight * resizedW];
        resizedFloatMat.get(0, 0, hwc);

        // 归一化 + HWC->CHW 交织，写入 out。
        // 通道外层使 out 按行连续写（hwc 跨步读，其驻留热 cache 代价低），
        // 归一化表达式与写入位置与旧逐像素写法完全一致（位不变）
        int plane = imgHeight * imgWidth;
        for (int c = 0; c < imgChannel; c++) {
            int cBase = offset + c * plane;
            for (int i = 0; i < imgHeight; i++) {
                int srcRow = (i * resizedW) * imgChannel + c;
                int dstRow = cBase + i * imgWidth;
                for (int j = 0; j < resizedW; j++) {
                    float normalizedValue = (hwc[srcRow + j * imgChannel] / 255.0f) * 2.0f - 1.0f;
                    out[dstRow + j] = normalizedValue;
                }
            }
        }

        resizedMat.release();
        resizedFloatMat.release();
    }

}
