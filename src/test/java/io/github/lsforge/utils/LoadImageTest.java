package io.github.lsforge.utils;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * LoadImage 输入转换工具单元测试。
 */
public class LoadImageTest {

    private static final String TEST_IMG = "src/test/resources/ch_en_num.jpg";

    private LoadImage loadImage;

    @BeforeAll
    static void init() {
        OpencvLoader.loadOpencvLib();
    }

    @org.junit.jupiter.api.BeforeEach
    void setUp() {
        loadImage = new LoadImage();
    }

    @Test
    void shouldLoadFromStringPath() throws Exception {
        Mat mat = loadImage.call(TEST_IMG);
        assertValidColorMat(mat);
    }

    @Test
    void shouldLoadFromJavaPath() throws Exception {
        Path path = Paths.get(TEST_IMG);
        Mat mat = loadImage.call(path);
        assertValidColorMat(mat);
    }

    @Test
    void shouldLoadFromByteArray() throws Exception {
        byte[] bytes = Files.readAllBytes(Paths.get(TEST_IMG));
        Mat mat = loadImage.call(bytes);
        assertValidColorMat(mat);
    }

    @Test
    void shouldPassThroughMat() throws Exception {
        Mat original = Mat.ones(100, 200, CvType.CV_8UC3);
        Mat result = loadImage.call(original);
        Assertions.assertSame(original, result, "Mat 输入应直接透传");
    }

    @Test
    void shouldLoadFromBufferedImage() throws Exception {
        BufferedImage image = ImageIO.read(new File(TEST_IMG));
        Mat mat = loadImage.call(image);
        assertValidColorMat(mat);
    }

    @Test
    void shouldRejectUnsupportedType() {
        Exception ex = Assertions.assertThrows(Exception.class, () -> loadImage.call(12345));
        Assertions.assertTrue(ex.getMessage().contains("输入类型不被支持"));
    }

    @Test
    void shouldConvertGrayscaleToBGR() throws Exception {
        Mat gray = Mat.ones(100, 200, CvType.CV_8UC1);
        Mat bgr = loadImage.call(gray);
        Assertions.assertEquals(3, bgr.channels(), "灰度图应被转换为 3 通道 BGR");
    }

    @Test
    void shouldConvertRGBAtoBGR() throws Exception {
        Mat rgba = Mat.ones(100, 200, CvType.CV_8UC4);
        Mat bgr = loadImage.call(rgba);
        Assertions.assertEquals(3, bgr.channels(), "RGBA 应被转换为 3 通道 BGR");
    }

    private static void assertValidColorMat(Mat mat) {
        Assertions.assertFalse(mat.empty(), "Mat 不应为空");
        Assertions.assertEquals(3, mat.channels(), "应为 BGR 三通道");
        Assertions.assertTrue(mat.rows() > 0 && mat.cols() > 0, "尺寸应大于 0");
    }
}
