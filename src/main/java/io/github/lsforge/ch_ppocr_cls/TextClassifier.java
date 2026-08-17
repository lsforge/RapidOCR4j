package io.github.lsforge.ch_ppocr_cls;

import ai.onnxruntime.OrtException;
import io.github.lsforge.entity.FlatTensor;
import io.github.lsforge.entity.OcrConfig;
import io.github.lsforge.entity.Pair;
import io.github.lsforge.entity.Triple;
import io.github.lsforge.entity.OrtInferConfig;
import io.github.lsforge.utils.OrtInferSession;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * 对输入图像进行文字方向分类，并在需要时旋转图像。
 */
public class TextClassifier implements AutoCloseable {

    // 分类输入图像的形状 [channels, height, width]
    private final int[] clsImageShape;
    // 分类批量处理数量
    private final int clsBatchNum;
    // 分类阈值
    private final float clsThresh;
    // 分类后处理
    private final ClsPostProcess postProcessOp;
    // ONNX 运行会话
    private final OrtInferSession inferSession;

    /**
     * 从 OCRConfig 中读取分类相关的配置，并进行初始化。
     *
     * @param clsConfig      OCRConfig，其中包含了分类模块 ClsConfig 的相关配置
     * @param ortInferConfig 推理会话配置（引擎参数来自 Global，由门面组装）
     */
    public TextClassifier(OcrConfig.ClsConfig clsConfig, OrtInferConfig ortInferConfig) {
        // 从配置中读取相关的分类参数
        this.clsImageShape = clsConfig.getClsImageShape(); // [C, H, W]
        this.clsBatchNum = clsConfig.getClsBatchNum();
        this.clsThresh = clsConfig.getClsThresh();
        this.postProcessOp = new ClsPostProcess(clsConfig.getLabelList());

        // 初始化推理会话
        this.inferSession = new OrtInferSession(ortInferConfig);
    }

    /**
     * 对输入图像列表进行分类，若识别到 180 度的标签且分数超过阈值，则进行图像旋转。
     *
     * @param imgList 待分类的图像列表（OpenCV Mat 格式）
     * @return 三元组：(1) 处理后的图像列表（可能会被旋转）；(2) [标签, 分数] 列表；(3) 运行所耗时(秒)
     */
    public Triple<List<Mat>, List<Pair<String, Float>>, Double> call(List<Mat> imgList) {
        // 记录开始时间
        long start = System.currentTimeMillis();

        // 计算每张图像的宽高比，用于排序优化
        List<Double> widthRatioList = new ArrayList<>();
        for (Mat img : imgList) {
            double h = img.rows();
            double w = img.cols();
            widthRatioList.add(w / h);
        }

        // 获取排序后的索引序列
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < imgList.size(); i++) {
            indices.add(i);
        }
        // 按照宽高比升序排序
        indices.sort(Comparator.comparingDouble(widthRatioList::get));

        // 准备存放分类结果 [label, score]
        List<Pair<String, Float>> clsRes = new ArrayList<>();
        for (int i = 0; i < imgList.size(); i++) {
            // 先填充空结果，保持位置对齐
            clsRes.add(Pair.of("", 0.0f));
        }

        int imgNum = imgList.size();

        // 分批处理
        for (int beg = 0; beg < imgNum; beg += clsBatchNum) {
            int end = Math.min(imgNum, beg + clsBatchNum);

            // 构建当前 batch 的扁平预处理结果 [n, C, H, W]
            int imgC = clsImageShape[0];
            int imgH = clsImageShape[1];
            int imgW = clsImageShape[2];
            int sampleSize = imgC * imgH * imgW;
            float[] batchData = new float[(end - beg) * sampleSize];

            for (int idx = beg; idx < end; idx++) {
                int realIndex = indices.get(idx);
                // resize & normalize，直接写入本样本在批次中的偏移位置
                resizeNormImg(imgList.get(realIndex), batchData, (idx - beg) * sampleSize);
            }

            // 推理
            FlatTensor probOut;
            try {
                probOut = inferSession.run(batchData, new long[]{end - beg, imgC, imgH, imgW});
            } catch (OrtException e) {
                e.printStackTrace();
                continue;
            }
            // 后处理，得到每个样本的 [label, score]（输出形状 [n, numClasses]）
            List<Pair<String, Float>> clsResult = postProcessOp.call(probOut.getData(), probOut.dim(1));

            // 将分类结果放回原来的顺序位置
            for (int rno = 0; rno < clsResult.size(); rno++) {
                Pair<String, Float> pair = clsResult.get(rno);
                String label = pair.getLeft();
                float score = pair.getRight();

                // 放到 clsRes 对应的真实索引位置中
                int realIndex = indices.get(beg + rno);
                clsRes.set(realIndex, Pair.of(label, score));

                // 如果预测为 “180” 且置信度超过阈值，则在原图上进行 180 度旋转
                if (label.contains("180") && score > clsThresh) {
                    // OpenCV 旋转 180 度
                    Mat original = imgList.get(realIndex);
                    Core.rotate(original, original, Core.ROTATE_180);
                }
            }
        }

        // 计算总耗时（秒）
        long end = System.currentTimeMillis();
        double totalSeconds = (end - start) / 1000.0;

        // 返回处理过后的图像列表、分类结果以及耗时
        return Triple.of(imgList, clsRes, totalSeconds);
    }

    @Override
    public void close() {
        if (inferSession != null) {
            inferSession.close();
        }
    }

    /**
     * 将单张图像进行 resize 和归一化处理，结果以 CHW 行主序写入 out 的 offset 偏移处。
     *
     * @param img    OpenCV Mat 格式的单张图像
     * @param out    批次扁平数据（[n, C, H, W] 中本样本的 CHW 段）
     * @param offset 本样本在 out 中的起始偏移
     */
    private void resizeNormImg(Mat img, float[] out, int offset) {
        // clsImageShape: [channels, imgH, imgW]
        int imgC = clsImageShape[0];
        int imgH = clsImageShape[1];
        int imgW = clsImageShape[2];

        int h = img.rows();
        int w = img.cols();
        float ratio = (float) w / (float) h;

        // 根据长宽比，判断 resize 后的宽度
        int resizedW = (int) Math.ceil(imgH * ratio);
        if (resizedW > imgW) {
            resizedW = imgW;
        }

        // 1) 先把图像 resize 到 (resizedW, imgH)
        Mat resizedMat = new Mat();
        Imgproc.resize(img, resizedMat, new Size(resizedW, imgH));

        // 模型为单通道输入时，BGR 转灰度
        if (imgC == 1 && resizedMat.channels() == 3) {
            Imgproc.cvtColor(resizedMat, resizedMat, Imgproc.COLOR_BGR2GRAY);
        }

        // 2) 转换为 float 类型，并除以 255
        resizedMat.convertTo(resizedMat, CvType.CV_32FC(resizedMat.channels()), 1.0 / 255.0);

        // 3) 减均值 0.5，再除以 0.5（Scalar 版本，避免每张图分配两个全尺寸常量 Mat）
        Scalar meanStd = resizedMat.channels() == 1 ? new Scalar(0.5)
                : new Scalar(0.5, 0.5, 0.5);
        Core.subtract(resizedMat, meanStd, resizedMat);
        Core.divide(resizedMat, meanStd, resizedMat);

        // 4) 在 (imgH, imgW) 的画布上进行右侧 padding
        //    创建一个 (imgH, imgW) 大小，通道数为 imgC，初始值为 0.0 的 Mat
        Mat paddingMat = new Mat(imgH, imgW, CvType.CV_32FC(resizedMat.channels()));
        paddingMat.setTo(new Scalar(0.0, 0.0, 0.0));

        // 将 resizedMat 拷贝到 paddingMat 的左上角
        Mat subArea = paddingMat.submat(0, imgH, 0, resizedW);
        resizedMat.copyTo(subArea);

        // 5) HWC -> CHW 写入 out：通道分离后每通道一次批量读取（一次 JNI 拷贝）
        int c = paddingMat.channels();
        int plane = imgH * imgW;
        float[] planeBuf = new float[plane];
        List<Mat> channels = new ArrayList<>(c);
        Core.split(paddingMat, channels);
        for (int ch = 0; ch < c; ch++) {
            Mat channelMat = channels.get(ch);
            channelMat.get(0, 0, planeBuf);
            System.arraycopy(planeBuf, 0, out, offset + ch * plane, plane);
            channelMat.release();
        }

        resizedMat.release();
        subArea.release();
        paddingMat.release();
    }

}
