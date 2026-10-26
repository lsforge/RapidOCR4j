package io.github.lsforge.ch_ppocr_det;

import io.github.lsforge.entity.FlatTensor;
import io.github.lsforge.utils.OpencvLoader;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * 对齐 Python f65c7da：limit_type=max 时 limitSideLen 直接生效，不做尺寸分档
 */
public class TextDetectorPreprocessTest {

    @BeforeAll
    static void setUp() {
        OpencvLoader.loadOpencvLib();
    }

    @Test
    void maxTypeShouldHonorConfiguredLimitSideLen() {
        // 旧分档会按 maxWh=2000 取 2000（不缩放），新行为必须缩到配置的 736
        DetPreProcess preprocess = TextDetector.getPreprocess(736, "max");
        Mat img = new Mat(100, 2000, CvType.CV_8UC3);
        FlatTensor tensor = preprocess.call(img);
        assertEquals(736, tensor.dim(3), "limit_type=max 时输出宽必须缩到配置的 limitSideLen");
        assertEquals(32, tensor.dim(2));
    }

    @Test
    void maxTypeShouldNotUpscaleSmallImages() {
        // max(400) <= 736 不放大，仅 32 对齐（HALF_EVEN：12.5->12、9.375->9）
        DetPreProcess preprocess = TextDetector.getPreprocess(736, "max");
        Mat img = new Mat(300, 400, CvType.CV_8UC3);
        FlatTensor tensor = preprocess.call(img);
        assertEquals(384, tensor.dim(3));
        assertEquals(288, tensor.dim(2));
    }

    @Test
    void minTypeShouldKeepExistingBehavior() {
        // 短边 100 < 200 放大 2 倍后 32 对齐
        DetPreProcess preprocess = TextDetector.getPreprocess(200, "min");
        Mat img = new Mat(100, 2000, CvType.CV_8UC3);
        FlatTensor tensor = preprocess.call(img);
        assertEquals(192, tensor.dim(2));
        assertEquals(4000, tensor.dim(3));
    }
}
