package io.github.lsforge.utils;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * 对齐 Python 7cfd3d9：字节输入按 EXIF Orientation 转正，映射与 PIL 一致
 */
public class ExifOrientationTest {

    /** 2x3 单通道基准图：[[0,1,2],[3,4,5]]，像素值 = row*3+col */
    private static Mat baseImage() {
        Mat m = new Mat(2, 3, CvType.CV_8UC1);
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 3; c++) {
                m.put(r, c, r * 3 + c);
            }
        }
        return m;
    }

    private static int[][] pixels(Mat m) {
        int[][] out = new int[m.rows()][m.cols()];
        for (int r = 0; r < m.rows(); r++) {
            for (int c = 0; c < m.cols(); c++) {
                out[r][c] = (int) m.get(r, c)[0];
            }
        }
        return out;
    }

    @BeforeAll
    static void setUp() {
        OpencvLoader.loadOpencvLib();
    }

    // ---------- parse：合成 EXIF 字节流 ----------

    /** 拼装带 EXIF Orientation 的最小 JPEG 字节（marker 链 + APP1 + EOI） */
    private static byte[] jpegWithOrientation(int orientation, boolean bigEndian) {
        byte[] tiff;
        if (bigEndian) {
            tiff = new byte[]{
                    'M', 'M', 0, 42, 0, 0, 0, 8,
                    0, 1,
                    0x01, 0x12, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, (byte) orientation, 0x00, 0x00
            };
        } else {
            tiff = new byte[]{
                    'I', 'I', 42, 0, 8, 0, 0, 0,
                    1, 0,
                    0x12, 0x01, 3, 0, 1, 0, 0, 0, (byte) orientation, 0, 0, 0
            };
        }
        byte[] payload = new byte[6 + tiff.length]; // "Exif\0\0" + TIFF
        payload[0] = 'E';
        payload[1] = 'x';
        payload[2] = 'i';
        payload[3] = 'f';
        System.arraycopy(tiff, 0, payload, 6, tiff.length);
        int segLen = 2 + payload.length;
        byte[] out = new byte[2 + 2 + payload.length + 2]; // SOI + APP1 头 + payload + EOI
        out[0] = (byte) 0xFF;
        out[1] = (byte) 0xD8;
        out[2] = (byte) 0xFF;
        out[3] = (byte) 0xE1;
        out[4] = (byte) (segLen >> 8);
        out[5] = (byte) segLen;
        System.arraycopy(payload, 0, out, 6, payload.length);
        out[out.length - 2] = (byte) 0xFF;
        out[out.length - 1] = (byte) 0xD9;
        return out;
    }

    @Test
    void shouldParseAllOrientationsBothEndians() {
        for (int v = 1; v <= 8; v++) {
            assertEquals(v, ExifOrientation.parse(jpegWithOrientation(v, false)), "小端 orientation=" + v);
            assertEquals(v, ExifOrientation.parse(jpegWithOrientation(v, true)), "大端 orientation=" + v);
        }
    }

    @Test
    void shouldReturnOneWithoutExif() {
        assertEquals(1, ExifOrientation.parse(null));
        assertEquals(1, ExifOrientation.parse(new byte[]{0x00, 0x01})); // 非 JPEG
        // 只有 SOI/EOI 的裸 JPEG
        assertEquals(1, ExifOrientation.parse(new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF, (byte) 0xD9}));
    }

    // ---------- apply：几何断言（基准图 [[0,1,2],[3,4,5]]） ----------

    @Test
    void applyShouldMatchPilTransposes() {
        // 6: 顺时针 90°（PIL ROTATE_270）
        Mat m6 = baseImage();
        ExifOrientation.apply(m6, 6);
        assertEquals(3, m6.rows());
        assertEquals(2, m6.cols());
        org.junit.jupiter.api.Assertions.assertArrayEquals(new int[][]{{3, 0}, {4, 1}, {5, 2}}, pixels(m6));

        // 8: 逆时针 90°（PIL ROTATE_90）
        Mat m8 = baseImage();
        ExifOrientation.apply(m8, 8);
        org.junit.jupiter.api.Assertions.assertArrayEquals(new int[][]{{2, 5}, {1, 4}, {0, 3}}, pixels(m8));

        // 3: 180°
        Mat m3 = baseImage();
        ExifOrientation.apply(m3, 3);
        org.junit.jupiter.api.Assertions.assertArrayEquals(new int[][]{{5, 4, 3}, {2, 1, 0}}, pixels(m3));

        // 2: 水平镜像；4: 垂直镜像
        Mat m2 = baseImage();
        ExifOrientation.apply(m2, 2);
        org.junit.jupiter.api.Assertions.assertArrayEquals(new int[][]{{2, 1, 0}, {5, 4, 3}}, pixels(m2));

        Mat m4 = baseImage();
        ExifOrientation.apply(m4, 4);
        org.junit.jupiter.api.Assertions.assertArrayEquals(new int[][]{{3, 4, 5}, {0, 1, 2}}, pixels(m4));

        // 5: 主对角转置；7: 副对角转置
        Mat m5 = baseImage();
        ExifOrientation.apply(m5, 5);
        org.junit.jupiter.api.Assertions.assertArrayEquals(new int[][]{{0, 3}, {1, 4}, {2, 5}}, pixels(m5));

        Mat m7 = baseImage();
        ExifOrientation.apply(m7, 7);
        org.junit.jupiter.api.Assertions.assertArrayEquals(new int[][]{{5, 2}, {4, 1}, {3, 0}}, pixels(m7));

        // 1/未知：不处理
        Mat m1 = baseImage();
        ExifOrientation.apply(m1, 1);
        org.junit.jupiter.api.Assertions.assertArrayEquals(new int[][]{{0, 1, 2}, {3, 4, 5}}, pixels(m1));
    }

    // ---------- 端到端：LoadImage byte[] 入口 ----------

    @Test
    void loadImageShouldTransposeJpegBytes(@TempDir Path tmpDir) throws Exception {
        // 6 行 4 列渐变图，写入真实 JPEG
        Mat src = new Mat(6, 4, CvType.CV_8UC1);
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 4; c++) {
                src.put(r, c, r * 10 + c);
            }
        }
        Path jpg = tmpDir.resolve("oriented.jpg");
        org.junit.jupiter.api.Assertions.assertTrue(Imgcodecs.imwrite(jpg.toString(), src));
        byte[] raw = Files.readAllBytes(jpg);

        // 无 EXIF：尺寸不变
        Mat plain = new LoadImage().call(raw);
        assertEquals(6, plain.rows());
        assertEquals(4, plain.cols());

        // 插入 orientation=6 的 APP1 段：顺时针 90° 转正后应为 4 行 6 列
        byte[] tagged = withExifApp1(raw, 6);
        Mat rotated = new LoadImage().call(tagged);
        assertEquals(4, rotated.rows());
        assertEquals(6, rotated.cols());
        // 顺时针 90°：原右下角像素 (5,3)=53 转到新左下角 (3,0)
        assertEquals(53.0, rotated.get(3, 0)[0], 1.0);
    }

    /** 在 JPEG 字节的 SOI 之后插入携带指定 Orientation 的 APP1 段（测试辅助） */
    private static byte[] withExifApp1(byte[] jpeg, int orientation) {
        byte[] tiff = new byte[]{
                'I', 'I', 42, 0, 8, 0, 0, 0,
                1, 0,
                0x12, 0x01, 3, 0, 1, 0, 0, 0, (byte) orientation, 0, 0, 0
        };
        byte[] payload = new byte[6 + tiff.length];
        payload[0] = 'E';
        payload[1] = 'x';
        payload[2] = 'i';
        payload[3] = 'f';
        System.arraycopy(tiff, 0, payload, 6, tiff.length);
        int segLen = 2 + payload.length;
        byte[] out = new byte[jpeg.length + 4 + payload.length];
        System.arraycopy(jpeg, 0, out, 0, 2); // SOI
        out[2] = (byte) 0xFF;
        out[3] = (byte) 0xE1;
        out[4] = (byte) (segLen >> 8);
        out[5] = (byte) segLen;
        System.arraycopy(payload, 0, out, 6, payload.length);
        System.arraycopy(jpeg, 2, out, 6 + payload.length, jpeg.length - 2);
        return out;
    }
}
