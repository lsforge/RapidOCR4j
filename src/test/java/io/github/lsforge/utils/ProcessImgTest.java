package io.github.lsforge.utils;

import io.github.lsforge.entity.Triple;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

/**
 * ProcessImg 图像处理工具单元测试。
 */
public class ProcessImgTest {

    @BeforeAll
    static void init() {
        OpencvLoader.loadOpencvLib();
    }

    @Test
    void alignTo32ShouldUseBankersRounding() {
        // 2000*1 = 2000; 2000/32 = 62.5，半值应舍入到偶数 62 -> 1984
        Assertions.assertEquals(1984, ProcessImg.alignTo32(2000, 1.0));
        // 2048*1 = 2048; 2048/32 = 64 -> 2048
        Assertions.assertEquals(2048, ProcessImg.alignTo32(2048, 1.0));
        // 2064/32 = 64.5，半值应舍入到偶数 64 -> 2048
        Assertions.assertEquals(2048, ProcessImg.alignTo32(2064, 1.0));
    }

    @Test
    void reduceMaxSideShouldScaleDownLargeImage() throws ProcessImg.ResizeImgError {
        Mat img = Mat.zeros(2000, 2000, org.opencv.core.CvType.CV_8UC3);
        Triple<Mat, Float, Float> result = ProcessImg.reduceMaxSide(img, 1000);

        Assertions.assertTrue(result.getLeft().rows() <= 1000, "高度应被缩小");
        Assertions.assertTrue(result.getLeft().cols() <= 1000, "宽度应被缩小");
        Assertions.assertTrue(result.getMiddle() > 1.0f, "高度缩放比应大于 1");
        Assertions.assertTrue(result.getRight() > 1.0f, "宽度缩放比应大于 1");
    }

    @Test
    void increaseMinSideShouldScaleUpSmallImage() throws ProcessImg.ResizeImgError {
        Mat img = Mat.zeros(100, 200, org.opencv.core.CvType.CV_8UC3);
        Triple<Mat, Float, Float> result = ProcessImg.increaseMinSide(img, 300);

        Assertions.assertTrue(result.getLeft().rows() >= 300 || result.getLeft().cols() >= 300,
                "至少有一条边被放大到目标值");
    }

    @Test
    void addRoundLetterboxShouldAddBlackBorder() {
        Mat img = Mat.ones(100, 200, org.opencv.core.CvType.CV_8UC3);
        Mat padded = ProcessImg.addRoundLetterbox(img, 10, 20, 5, 15);

        Assertions.assertEquals(130, padded.rows(), "高度应增加 top+bottom");
        Assertions.assertEquals(220, padded.cols(), "宽度应增加 left+right");

        Scalar mean = Core.mean(padded);
        Assertions.assertTrue(mean.val[0] < 255, "填充区域为黑色，平均值应小于全白");
    }
}
