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
    void shouldReadSubimagePixelsNotParentCorner() throws Exception {
        // 构造一幅内容随位置变化的 3BYTE_BGR 父图，确保不同区域的像素必然不同
        int pw = 320, ph = 240;
        BufferedImage parent = new BufferedImage(pw, ph, BufferedImage.TYPE_3BYTE_BGR);
        byte[] parentData = new byte[pw * ph * 3];
        for (int y = 0; y < ph; y++) {
            for (int x = 0; x < pw; x++) {
                int base = (y * pw + x) * 3;
                parentData[base] = (byte) ((x * 7) & 0xFF);
                parentData[base + 1] = (byte) ((y * 5) & 0xFF);
                parentData[base + 2] = (byte) ((x + y) & 0xFF);
            }
        }
        parent.getRaster().setDataElements(0, 0, pw, ph, parentData);

        int sx = 7, sy = 11, sw = 200, sh = 100;
        BufferedImage sub = parent.getSubimage(sx, sy, sw, sh);
        // 前提校验：子图与父图同类型，且数据区大于子图（共享 DataBuffer）
        Assertions.assertEquals(BufferedImage.TYPE_3BYTE_BGR, sub.getType());
        Assertions.assertTrue(parent.getRaster().getDataBuffer().getSize() > sw * sh * 3);

        Mat mat = loadImage.call(sub);
        assertValidColorMat(mat);
        Assertions.assertEquals(sh, mat.rows());
        Assertions.assertEquals(sw, mat.cols());

        // 期望值：从父图 DataBuffer 读回实际内存内容，取 (sx, sy) 起步的子区域
        byte[] parentMemory = ((java.awt.image.DataBufferByte) parent.getRaster().getDataBuffer()).getData();
        byte[] expectedData = new byte[sw * sh * 3];
        for (int y = 0; y < sh; y++) {
            System.arraycopy(parentMemory, ((sy + y) * pw + sx) * 3, expectedData, y * sw * 3, sw * 3);
        }
        byte[] actualData = new byte[sw * sh * 3];
        mat.get(0, 0, actualData);
        Assertions.assertArrayEquals(expectedData, actualData, "子图转 Mat 必须等于父图对应裁剪区域，而不是父图左上角");
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
