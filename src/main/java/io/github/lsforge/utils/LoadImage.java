package io.github.lsforge.utils;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * 自定义异常类
 */
class LoadImageError extends Exception {
    public LoadImageError(String message) {
        super(message);
    }

    public LoadImageError(String message, Throwable cause) {
        super(message, cause);
    }
}

/**
 * 将输入图片转换为 BGR 格式的 Mat。
 * String/Path/byte[] 输入按 EXIF Orientation 转正（对齐 Python exif_transpose）；Mat/BufferedImage 不处理
 */
public class LoadImage {

    /**
     * 核心方法
     *
     * @param img 可能是：
     *            1. String/Path: 图片路径
     *            2. byte[]: 图片文件的二进制数据
     *            3. Mat: 已经是OpenCV的Mat对象
     *            4. BufferedImage: 图片对象
     * @return OpenCV 的 Mat，通道顺序为 BGR
     * @throws LoadImageError 当加载或处理失败时，抛出异常
     */
    public Mat call(Object img) throws LoadImageError {
        if (!(img instanceof String || img instanceof Path || img instanceof byte[]
                || img instanceof Mat || img instanceof BufferedImage)) {
            throw new LoadImageError("输入类型不被支持: " + img.getClass().getName());
        }

        Mat mat = loadImg(img);

        Mat converted = convertImg(mat);

        // 如果 convertImg 创建了新的 Mat，释放原始 Mat
        if (converted != mat) {
            mat.release();
        }

        return converted;
    }

    /**
     * 加载图片，转为 OpenCV 的 Mat 格式
     *
     * @param img 输入可能是多种类型
     * @return Mat
     * @throws LoadImageError 加载失败抛出异常
     */
    private Mat loadImg(Object img) throws LoadImageError {
        // 1. 字符串/Path：统一读字节走 imdecode（兼容中文路径，且可配合 EXIF 转正）
        if (img instanceof String || img instanceof Path) {
            String filePath = img instanceof String ? (String) img : ((Path) img).toString();
            verifyExist(filePath);
            byte[] bytes;
            try {
                bytes = Files.readAllBytes(Paths.get(filePath));
            } catch (IOException e) {
                throw new LoadImageError("无法识别或读取图片: " + filePath);
            }
            return decodeWithExif(bytes, "无法识别或读取图片: " + filePath);
        }

        // 2. 如果是 byte[]，认为是图片的二进制内容
        if (img instanceof byte[]) {
            return decodeWithExif((byte[]) img, "无法识别或读取二进制图片数据");
        }

        // 3. 如果已经是 Mat，则直接返回
        if (img instanceof Mat) {
            return (Mat) img;
        }

        // 4. 如果是 BufferedImage 转 Mat
        if (img instanceof BufferedImage) {
            return bufferedImageToMat((BufferedImage) img);
        }

        // 5. 其他类型不支持
        throw new LoadImageError("不支持的图片输入类型: " + img.getClass().getName());
    }

    /** imdecode 解码并按 EXIF Orientation 转正（解析失败按无 EXIF 处理） */
    private static Mat decodeWithExif(byte[] bytes, String errorMessage) throws LoadImageError {
        MatOfByte mob = new MatOfByte(bytes);
        // IMREAD_IGNORE_ORIENTATION 禁用 OpenCV 隐式 EXIF 转正，统一走下方显式映射（避免二次转正）
        Mat mat = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR | Imgcodecs.IMREAD_IGNORE_ORIENTATION);
        if (mat.empty()) {
            throw new LoadImageError(errorMessage);
        }
        ExifOrientation.apply(mat, ExifOrientation.parse(bytes));
        return mat;
    }

    /**
     * 将图像转换为 BGR 三通道
     *
     * 注意：imread/imdecode(IMREAD_COLOR) 与 bufferedImageToMat 产出的 Mat 本身就是 BGR 顺序，
     * 此处无需再做 RGB2BGR 转换（对齐 Python 端 imdecode 后直接使用的行为）
     *
     * @param img OpenCV 的 Mat（可能是多通道）
     * @return BGR 格式的 Mat
     * @throws LoadImageError 如果通道数/维度异常
     */
    private Mat convertImg(Mat img) throws LoadImageError {
        // OpenCV 的 Mat 通常是 2D（图像高度、宽度），通道数可以从 type 或者 shape 获得
        int channels = img.channels();
        int depth = img.depth();
        int rows = img.rows();
        int cols = img.cols();

        // 如果是单通道灰度
        if (channels == 1) {
            // 将灰度图转换为 BGR
            Mat bgrMat = new Mat();
            Imgproc.cvtColor(img, bgrMat, Imgproc.COLOR_GRAY2BGR);
            return bgrMat;
        }

        // 如果是两通道 (例如灰度 + alpha)
        if (channels == 2) {
            return cvtTwoToThree(img);
        }

        // 如果是三通道（imread/imdecode/BufferedImage 转换产物均为 BGR），直接返回
        if (channels == 3) {
            return img;
        }

        // 如果是四通道 (例如 RGBA)
        if (channels == 4) {
            return cvtFourToThree(img);
        }

        // 如果通道数不在 [1, 2, 3, 4]，则抛出异常
        throw new LoadImageError("图像通道数(" + channels + ")不在[1, 2, 3, 4]范围内！");
    }

    /**
     * 将灰度+alpha 的两通道图转换为 BGR 三通道
     *
     * @param img 两通道图像
     * @return 转换后得到的 BGR
     */
    private Mat cvtTwoToThree(Mat img) {
        // 拆分通道: [灰度, alpha]
        List<Mat> channels = new ArrayList<>();
        Core.split(img, channels);

        // 第一个通道：灰度
        Mat gray = channels.get(0);
        // 第二个通道：alpha
        Mat alpha = channels.get(1);

        // 将灰度转换为 BGR
        Mat bgr = new Mat();
        Imgproc.cvtColor(gray, bgr, Imgproc.COLOR_GRAY2BGR);

        // alpha 取反（类似 bitwise_not）
        Mat notAlpha = new Mat();
        Core.bitwise_not(alpha, notAlpha);

        // 将 notAlpha 也转换为三通道，便于与 BGR 做后续操作
        Mat notAlpha3 = new Mat();
        Imgproc.cvtColor(notAlpha, notAlpha3, Imgproc.COLOR_GRAY2BGR);

        // 使用 alpha 作为掩膜，对原 BGR 部分进行保留
        Mat newImg = new Mat();
        Core.bitwise_and(bgr, bgr, newImg, alpha);

        // 将 newImg 与 notAlpha3 合并
        Core.add(newImg, notAlpha3, newImg);

        // 释放中间 Mat 头（split 出的通道与输入共享数据，只需释放头）
        gray.release();
        alpha.release();
        bgr.release();
        notAlpha.release();
        notAlpha3.release();

        return newImg;
    }

    /**
     * 将 RGBA 的四通道图转换为 BGR 三通道
     *
     * @param img 四通道 RGBA
     * @return 转换后得到的 BGR
     */
    private Mat cvtFourToThree(Mat img) {
        // 拆分 RGBA
        List<Mat> channels = new ArrayList<>();
        Core.split(img, channels); // [R, G, B, A]

        Mat r = channels.get(0);
        Mat g = channels.get(1);
        Mat b = channels.get(2);
        Mat a = channels.get(3);

        // 合并 BGR (在 OpenCV 中通常顺序是 B, G, R)
        // 由于 Python 逻辑假定输入是 RGBA，因此这里顺序可做调整
        // 若原图确实是 RGBA，则 channels[0] = R, channels[1] = G, channels[2] = B
        // 这里想要得到 BGR 需要 merge(b, g, r)
        List<Mat> bgrList = new ArrayList<>();
        bgrList.add(b);
        bgrList.add(g);
        bgrList.add(r);
        Mat bgr = new Mat();
        Core.merge(bgrList, bgr);

        // alpha 取反
        Mat notA = new Mat();
        Core.bitwise_not(a, notA);

        // 转成三通道
        Mat notA3 = new Mat();
        Imgproc.cvtColor(notA, notA3, Imgproc.COLOR_GRAY2BGR);

        // 使用 alpha 作为掩膜，对 BGR 进行保留
        Mat newImg = new Mat();
        Core.bitwise_and(bgr, bgr, newImg, a);

        // 计算 newImg 的平均值（类似 np.mean(new_img)）
        // meanScalar.val[0], [1], [2], [3] 对应 B, G, R, alpha
        Scalar meanScalar = Core.mean(newImg);
        double avgColor = (meanScalar.val[0] + meanScalar.val[1] + meanScalar.val[2]) / 3.0;

        Mat result;
        if (avgColor <= 0.0) {
            // 如果平均值接近 0，说明基本是黑色，则将 newImg 与 notA3 合并
            Core.add(newImg, notA3, newImg);
            result = newImg;
        } else {
            // 否则对 newImg 进行取反，得到类似黑色背景的效果
            Mat inverted = new Mat();
            Core.bitwise_not(newImg, inverted);
            result = inverted;
            newImg.release();
        }

        // 释放中间 Mat 头
        r.release();
        g.release();
        b.release();
        a.release();
        bgr.release();
        notA.release();
        notA3.release();

        return result;
    }

    /**
     * 验证文件是否存在
     *
     * @param filePath 文件路径
     * @throws LoadImageError 不存在时抛出异常
     */
    private void verifyExist(String filePath) throws LoadImageError {
        File f = new File(filePath);
        if (!f.exists() || !f.isFile()) {
            throw new LoadImageError("文件不存在: " + filePath);
        }
    }

    /**
     * 将 BufferedImage 转为 Mat
     * @param bi 传入的 BufferedImage
     * @return 转换后的 Mat
     */
    private Mat bufferedImageToMat(BufferedImage bi) {
        // 仅当类型为 TYPE_3BYTE_BGR 且数据区与图像严格等大时才走零拷贝快路径。
        // getSubimage 的子图与父图共享 DataBuffer（数据区比子图大），
        // 若直接读取，裁剪偏移会丢失：Mat.put 静默截断后灌入的是父图左上角内容，
        // 导致识别内容错位甚至返回空结果，因此这类图像必须重绘为独立图像。
        if (!canReadDirectly(bi)) {
            // 先转换为 TYPE_3BYTE_BGR 类型（OpenCV 默认是 BGR）
            BufferedImage convertedImg = new BufferedImage(bi.getWidth(), bi.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
            Graphics2D g = convertedImg.createGraphics();
            g.drawImage(bi, 0, 0, null);
            g.dispose();
            bi = convertedImg;
        }

        byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
        Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
        mat.put(0, 0, data);
        return mat;
    }

    /**
     * 判断能否直接读取 raster 数据区（零拷贝快路径）：
     * 类型为 TYPE_3BYTE_BGR，且 DataBuffer 大小与 w*h*3 严格相等
     * （子图/带行步距的 raster 均不满足，走重绘慢路径）
     */
    private static boolean canReadDirectly(BufferedImage bi) {
        if (bi.getType() != BufferedImage.TYPE_3BYTE_BGR) {
            return false;
        }
        DataBuffer dataBuffer = bi.getRaster().getDataBuffer();
        return dataBuffer instanceof DataBufferByte
                && dataBuffer.getSize() == bi.getWidth() * bi.getHeight() * 3;
    }

}

