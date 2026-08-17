package io.github.lsforge;

import io.github.lsforge.entity.OcrConfig;
import io.github.lsforge.entity.OcrResult;
import io.github.lsforge.entity.ParamConfig;
import io.github.lsforge.entity.RecResult;
import io.github.lsforge.utils.OpencvLoader;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

/**
 * OCR 端到端集成测试：验证 RapidOCR 对多种输入类型、调用参数及预处理开关的处理。
 * 所有测试共享同一个 RapidOCR 实例，避免重复加载 ONNX 模型。
 */
public class OcrIntegrationTest {

    private static final String TEST_IMG_TEXT = "src/test/resources/ch_en_num.jpg";

    private static RapidOCR sharedEngine;

    @BeforeAll
    static void init() {
        OpencvLoader.loadOpencvLib();
        sharedEngine = RapidOCR.create();
    }

    @Test
    void shouldRecognizeFromStringPath() throws Exception {
        OcrResult result = sharedEngine.run(TEST_IMG_TEXT);
        assertRecognized(result);
    }

    @Test
    void shouldRecognizeFromJavaPath() throws Exception {
        Path path = Paths.get(TEST_IMG_TEXT);
        OcrResult result = sharedEngine.run(path);
        assertRecognized(result);
    }

    @Test
    void shouldRecognizeFromByteArray() throws Exception {
        byte[] bytes = Files.readAllBytes(Paths.get(TEST_IMG_TEXT));
        OcrResult result = sharedEngine.run(bytes);
        assertRecognized(result);
    }

    @Test
    void shouldRecognizeFromBufferedImage() throws Exception {
        BufferedImage image = ImageIO.read(new File(TEST_IMG_TEXT));
        OcrResult result = sharedEngine.run(image);
        assertRecognized(result);
    }

    @Test
    void shouldRecognizeFromMat() throws Exception {
        Mat mat = Imgcodecs.imread(new File(TEST_IMG_TEXT).getAbsolutePath());
        OcrResult result = sharedEngine.run(mat);
        assertRecognized(result);
    }

    @Test
    void shouldRecognizeChinesePath() throws Exception {
        File src = new File(TEST_IMG_TEXT);
        File dst = new File("target/测试图片_中文.png");
        dst.getParentFile().mkdirs();
        Files.copy(src.toPath(), dst.toPath(), StandardCopyOption.REPLACE_EXISTING);

        OcrResult result = sharedEngine.run(dst.getAbsolutePath());
        assertRecognized(result);
    }

    @Test
    void higherTextScoreShouldNotIncreaseResultCount() throws Exception {
        OcrResult defaultResult = sharedEngine.run(TEST_IMG_TEXT);
        Assertions.assertFalse(defaultResult.getRecRes().isEmpty(), "默认阈值应识别到文本");

        ParamConfig highScore = new ParamConfig();
        highScore.setTextScore(0.99f);
        OcrResult strictResult = sharedEngine.run(TEST_IMG_TEXT, highScore);

        Assertions.assertTrue(
                strictResult.getRecRes().size() <= defaultResult.getRecRes().size(),
                "提高置信度阈值后，结果行数不应超过默认阈值"
        );
    }

    @Test
    void disablingPreprocessShouldStillReturnInBoundsBoxes() throws Exception {
        OcrConfig config = new OcrConfig();
        config.Global.setUsePreprocessImg(false);
        config.Global.setUseVerticalPadding(false);
        RapidOCR engine = RapidOCR.create(config);

        OcrResult result = engine.run(TEST_IMG_TEXT);
        assertRecognized(result);

        Mat img = Imgcodecs.imread(TEST_IMG_TEXT);
        for (RecResult rec : result.getRecRes()) {
            if (rec.getDtBoxes() == null) {
                continue;
            }
            for (org.opencv.core.Point p : rec.getDtBoxes()) {
                Assertions.assertTrue(p.x >= 0 && p.x <= img.cols(), "x 越界: " + p.x);
                Assertions.assertTrue(p.y >= 0 && p.y <= img.rows(), "y 越界: " + p.y);
            }
        }
    }

    @Test
    void shouldReturnWordBoxesWhenEnabled() throws Exception {
        ParamConfig param = new ParamConfig();
        param.setReturnWordBox(true);
        param.setUseCls(false);

        OcrResult result = sharedEngine.run(TEST_IMG_TEXT, param);
        assertRecognized(result);

        boolean hasWordBoxes = result.getRecRes().stream()
                .anyMatch(r -> r.getWordBoxResult() != null
                        && r.getWordBoxResult().getSortedWordBoxList() != null
                        && !r.getWordBoxResult().getSortedWordBoxList().isEmpty());
        Assertions.assertTrue(hasWordBoxes, "启用 returnWordBox 后应返回词级框");
    }

    private static void assertRecognized(OcrResult result) {
        Assertions.assertNotNull(result, "结果不应为 null");
        Assertions.assertFalse(result.getRecRes().isEmpty(), "应识别到至少一行文本");
        for (RecResult rec : result.getRecRes()) {
            Assertions.assertNotNull(rec.getText(), "文本不应为 null");
            Assertions.assertTrue(rec.getConfidence() > 0, "置信度应大于 0");
        }
    }
}
