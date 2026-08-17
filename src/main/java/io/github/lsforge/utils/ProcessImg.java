package io.github.lsforge.utils;

import io.github.lsforge.entity.Triple;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class ProcessImg {

    // 缩小图像到最大边长 max_side_len
    public static Triple<Mat, Float, Float> reduceMaxSide(Mat img, int maxSideLen) throws ResizeImgError {
        return resizeToSide(img, maxSideLen, true);
    }

    // 放大图像到最小边长 min_side_len
    public static Triple<Mat, Float, Float> increaseMinSide(Mat img, int minSideLen) throws ResizeImgError {
        return resizeToSide(img, minSideLen, false);
    }

    /**
     * 按边长限制等比缩放：useMax=true 时超最大边才缩小，否则低于最小边才放大。
     * 缩放比基于触发边的实际尺寸，目标尺寸对齐 32 的倍数。
     */
    private static Triple<Mat, Float, Float> resizeToSide(Mat img, int sideLen, boolean useMax) throws ResizeImgError {
        int h = img.rows();
        int w = img.cols();

        double ratio = 1.0;
        boolean trigger = useMax ? Math.max(h, w) > sideLen : Math.min(h, w) < sideLen;
        if (trigger) {
            // 取触发边（max 模式为较长边、min 模式为较短边）计算缩放比
            ratio = useMax ? (double) sideLen / Math.max(h, w) : (double) sideLen / Math.min(h, w);
        }

        int resizeH = alignTo32(h, ratio);
        int resizeW = alignTo32(w, ratio);

        if (resizeW <= 0 || resizeH <= 0) {
            throw new ResizeImgError("resize_w 或 resize_h 小于等于 0");
        }

        Mat resizedImg = new Mat();
        Imgproc.resize(img, resizedImg, new Size(resizeW, resizeH));

        float ratioH = (float) h / resizeH;
        float ratioW = (float) w / resizeW;

        return Triple.of(resizedImg, ratioH, ratioW);
    }

    /**
     * 复刻 Python 端的 32 对齐公式：int(h*ratio) 先截断，再 round(x/32)*32（银行家舍入）。
     * 注意不能用 Math.round（half-up）：Python round 是 half-even，且 2000/32=62.5 等半值
     * 在 max_side_len=2000 的常见路径上必然出现，half-up 会多出一档 32px 导致与 Python 结果错位。
     */
    public static int alignTo32(double len, double ratio) {
        int resized = (int) (len * ratio); // 向零截断，等价 Python int()
        return BigDecimal.valueOf(resized / 32.0)
                .setScale(0, RoundingMode.HALF_EVEN)
                .intValue() * 32;
    }

    // 添加带圆角边框的图像填充
    public static Mat addRoundLetterbox(Mat img, int top, int bottom, int left, int right) {
        Mat paddedImg = new Mat();
        Core.copyMakeBorder(img, paddedImg, top, bottom, left, right, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
        return paddedImg;
    }

    // 自定义异常类，用于捕获调整失败的情况
    public static class ResizeImgError extends Exception {
        public ResizeImgError(String message) {
            super(message);
        }
    }

}
