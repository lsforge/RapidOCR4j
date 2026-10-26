package io.github.lsforge.ch_ppocr_cls;

import io.github.lsforge.utils.OpencvLoader;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * 对齐 Python 8ac45f4：v2.0 cls 模型（PP-OCRv6 套件沿用）长行裁中段，短行原样返回
 */
public class TextClassifierCropTest {

    @BeforeAll
    static void setUp() {
        OpencvLoader.loadOpencvLib();
    }

    @Test
    void longLineShouldCropToModelAspectRatioMiddle() {
        // 48x960 长行，模型 48x192：keepW = ceil(48*192/48) = 192，裁水平居中段 [384, 576)
        Mat img = new Mat(48, 960, CvType.CV_8UC1);
        img.setTo(new Scalar(0));
        img.colRange(384, 576).setTo(new Scalar(255));

        Mat cropped = TextClassifier.cropLongLineMiddle(img, 48, 192);

        assertEquals(192, cropped.cols());
        assertEquals(48, cropped.rows());
        // 裁出的必须是中段：首列为白，裁剪窗口外的左端为黑
        assertEquals(255.0, cropped.get(24, 0)[0]);
        assertEquals(0.0, img.get(24, 0)[0]);
    }

    @Test
    void shortLineShouldStayUntouched() {
        // 100x400：keepW = ceil(100*192/48) = 400，不超宽，应原样返回（同一引用，不拷贝）
        Mat img = new Mat(100, 400, CvType.CV_8UC1);
        assertSame(img, TextClassifier.cropLongLineMiddle(img, 48, 192));
    }
}
