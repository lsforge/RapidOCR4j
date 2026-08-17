package io.github.lsforge.ch_ppocr_det;

import io.github.lsforge.entity.FlatTensor;
import io.github.lsforge.utils.ProcessImg;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * 图像检测前处理（使用 Scalar 来减均值、除标准差）
 */
public class DetPreProcess {
    // 均值、标准差、缩放因子
    private final Scalar meanScalar;
    private final Scalar stdScalar;
    private final double scale;

    // 限制边长与限制类型
    private final int limitSideLen;
    private final String limitType;

    /**
     * 构造函数，传入限制边长与限制类型（"max"或"min"），默认值可自行定义
     * @param limitSideLen 限制边长
     * @param limitType 限制类型
     */
    public DetPreProcess(int limitSideLen, String limitType) {
        // 与 Python: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], scale=1/255.0
        this.meanScalar = new Scalar(0.5, 0.5, 0.5);
        this.stdScalar = new Scalar(0.5, 0.5, 0.5);
        this.scale = 1.0 / 255.0;

        this.limitSideLen = limitSideLen;
        // 若未指定则默认用 "min"
        this.limitType = (limitType == null ? "min" : limitType);
    }

    /**
     * 对图像进行预处理: 1) resize  2) normalize  3) permute(HWC->CHW)  4) expandDims
     * 直接产出 [1, C, H, W] 的扁平张量（行主序），不再构建嵌套数组。
     *
     * @param img OpenCV Mat 格式图像，通常为 BGR
     * @return 扁平张量 [1, C, H, W]
     */
    public FlatTensor call(Mat img) {
        // 1. resize
        Mat resizedImg = resize(img);
        if (resizedImg == null) {
            // 若 resize 后尺寸 <=0，返回 null
            return null;
        }

        // 2. normalize
        //   (pixel * scale - mean) / std，用 Scalar 与 Core API 实现
        Mat normImg = normalize(resizedImg);

        // 3+4. permute (H,W,C)->(C,H,W) 并展开为 [1, C, H, W] 扁平数据
        //    通道分离 + 每通道一次批量读取，替代逐像素 get(row, col) 的 JNI 调用
        FlatTensor tensor = permuteToFlat(normImg);

        normImg.release();
        return tensor;
    }

    /**
     * resize 到 32 的倍数，
     * 若 limitType="max"，当 max(h,w)>limitSideLen 时才缩放；
     * 若 limitType="min"，当 min(h,w)<limitSideLen 时才放大；
     */
    private Mat resize(Mat img) {
        int h = img.rows();
        int w = img.cols();

        double ratio;
        if ("max".equalsIgnoreCase(limitType)) {
            // 若 max(h,w) > limitSideLen => ratio=limitSideLen/max(h,w) 否则 1.0
            if (Math.max(h, w) > limitSideLen) {
                ratio = (h > w) ? (double) limitSideLen / h : (double) limitSideLen / w;
            } else {
                ratio = 1.0;
            }
        } else {
            // 若 min(h,w) < limitSideLen => ratio=limitSideLen/min(h,w) 否则 1.0
            if (Math.min(h, w) < limitSideLen) {
                ratio = (h < w) ? (double) limitSideLen / h : (double) limitSideLen / w;
            } else {
                ratio = 1.0;
            }
        }

        // 对齐 Python：int(h*ratio) 截断 + round(x/32)*32 银行家舍入；下限保护 32 防极小图
        int resizeH = Math.max(ProcessImg.alignTo32(h, ratio), 32);
        int resizeW = Math.max(ProcessImg.alignTo32(w, ratio), 32);

        // 使用 OpenCV 进行 resize
        Mat dst = new Mat();
        try {
            Imgproc.resize(img, dst, new Size(resizeW, resizeH));
        } catch (Exception e) {
            throw new ResizeImgError("Fail to resize image", e);
        }
        return dst;
    }

    /**
     * normalize：
     * 先将图像类型转为 CV_32FC3，并在 convertTo(...)时乘以 scale
     * 再用 Scalar 减去 mean，除以 std
     */
    private Mat normalize(Mat img) {
        // 转成 float32 并乘以 scale，再以 Scalar 减均值、除标准差
        img.convertTo(img, CvType.CV_32FC3, scale);
        Core.subtract(img, meanScalar, img);
        Core.divide(img, stdScalar, img);

        return img;
    }

    /**
     * 将 (H,W,C) 交织存储的 Mat 转为 [1, C, H, W] 行主序扁平张量。
     * Core.split 原生分离通道后，每通道一次批量 get（JNI 拷贝），
     * 避免 H*W 次逐像素 img.get(row, col) 调用与逐像素 double[] 分配。
     */
    private FlatTensor permuteToFlat(Mat img) {
        int c = img.channels();
        int h = img.rows();
        int w = img.cols();
        int plane = h * w;

        float[] chw = new float[c * plane];
        float[] planeBuf = new float[plane];

        List<Mat> channels = new ArrayList<>(c);
        Core.split(img, channels);
        for (int ch = 0; ch < c; ch++) {
            Mat channelMat = channels.get(ch);
            channelMat.get(0, 0, planeBuf);
            System.arraycopy(planeBuf, 0, chw, ch * plane, plane);
            channelMat.release();
        }

        return new FlatTensor(chw, new long[]{1, c, h, w});
    }

    /**
     * 自定义异常，与 Python ResizeImgError 对应
     */
    public static class ResizeImgError extends RuntimeException {
        public ResizeImgError(String message, Throwable cause) {
            super(message, cause);
        }
    }

}
