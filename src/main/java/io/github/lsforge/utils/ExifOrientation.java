package io.github.lsforge.utils;

import org.opencv.core.Core;
import org.opencv.core.Mat;

/**
 * JPEG EXIF Orientation 解析与转正，映射与 PIL exif_transpose 一致（对齐 Python 7cfd3d9）
 */
public final class ExifOrientation {

    private ExifOrientation() {
    }

    /**
     * 解析 EXIF Orientation（1-8）；非 JPEG、无 EXIF 或解析异常返回 1（不转正）
     */
    public static int parse(byte[] data) {
        try {
            if (data == null || data.length < 4
                    || (data[0] & 0xFF) != 0xFF || (data[1] & 0xFF) != 0xD8) {
                return 1; // 非 JPEG（SOI 缺失）
            }
            int i = 2;
            while (i + 4 <= data.length) {
                if ((data[i] & 0xFF) != 0xFF) {
                    return 1; // marker 失步
                }
                int marker = data[i + 1] & 0xFF;
                if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD9)) {
                    i += 2; // 无长度字段的 marker（TEM/RST/SOI/EOI）
                    continue;
                }
                int segLen = ((data[i + 2] & 0xFF) << 8) | (data[i + 3] & 0xFF);
                if (segLen < 2) {
                    return 1;
                }
                if (marker == 0xDA) {
                    return 1; // SOS：后续为压缩数据
                }
                if (marker == 0xE1 && segLen >= 8
                        && data[i + 4] == 'E' && data[i + 5] == 'x' && data[i + 6] == 'i' && data[i + 7] == 'f'
                        && data[i + 8] == 0 && data[i + 9] == 0) {
                    return parseTiff(data, i + 10);
                }
                i += 2 + segLen;
            }
            return 1;
        } catch (RuntimeException e) {
            return 1;
        }
    }

    /**
     * 从 TIFF 头（"II"/"MM"）开始解析 IFD0，取 Orientation(0x0112) 标签值。
     */
    private static int parseTiff(byte[] d, int tiff) {
        if (tiff + 8 > d.length) {
            return 1;
        }
        boolean bigEndian;
        if (d[tiff] == 'I' && d[tiff + 1] == 'I') {
            bigEndian = false;
        } else if (d[tiff] == 'M' && d[tiff + 1] == 'M') {
            bigEndian = true;
        } else {
            return 1;
        }
        int ifd = tiff + u4(d, tiff + 4, bigEndian); // 头部：字节序 2B + magic 2B，IFD offset 从第 5 字节起
        if (ifd + 2 > d.length) {
            return 1;
        }
        int count = u2(d, ifd, bigEndian);
        for (int k = 0; k < count; k++) {
            int e = ifd + 2 + k * 12;
            if (e + 12 > d.length) {
                return 1;
            }
            if (u2(d, e, bigEndian) == 0x0112) {
                int v = u2(d, e + 8, bigEndian); // SHORT 值内联在 value 字段
                return (v >= 1 && v <= 8) ? v : 1;
            }
        }
        return 1;
    }

    private static int u2(byte[] d, int off, boolean bigEndian) {
        return bigEndian
                ? ((d[off] & 0xFF) << 8) | (d[off + 1] & 0xFF)
                : ((d[off + 1] & 0xFF) << 8) | (d[off] & 0xFF);
    }

    private static int u4(byte[] d, int off, boolean bigEndian) {
        return bigEndian
                ? ((d[off] & 0xFF) << 24) | ((d[off + 1] & 0xFF) << 16) | ((d[off + 2] & 0xFF) << 8) | (d[off + 3] & 0xFF)
                : ((d[off + 3] & 0xFF) << 24) | ((d[off + 2] & 0xFF) << 16) | ((d[off + 1] & 0xFF) << 8) | (d[off] & 0xFF);
    }

    /**
     * 按 Orientation 原地转正 Mat（与 PIL 一致）：
     * 2 水平镜像 / 3 旋转180° / 4 垂直镜像 / 5 主对角转置 / 6 顺时针90° / 7 副对角转置 / 8 逆时针90°，其余不处理
     */
    public static void apply(Mat mat, int orientation) {
        switch (orientation) {
            case 2:
                Core.flip(mat, mat, 1);
                break;
            case 3:
                Core.rotate(mat, mat, Core.ROTATE_180);
                break;
            case 4:
                Core.flip(mat, mat, 0);
                break;
            case 5: {
                // transpose 尺寸变化，经临时 Mat 回写
                Mat dst = new Mat();
                Core.transpose(mat, dst);
                dst.copyTo(mat);
                dst.release();
                break;
            }
            case 6:
                Core.rotate(mat, mat, Core.ROTATE_90_CLOCKWISE);
                break;
            case 7:
                Core.rotate(mat, mat, Core.ROTATE_90_CLOCKWISE);
                Core.flip(mat, mat, 0);
                break;
            case 8:
                Core.rotate(mat, mat, Core.ROTATE_90_COUNTERCLOCKWISE);
                break;
            default:
                break;
        }
    }
}
