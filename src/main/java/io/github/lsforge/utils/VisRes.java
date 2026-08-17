package io.github.lsforge.utils;

import io.github.lsforge.entity.RecResult;
import io.github.lsforge.entity.WordBoxResult;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class VisRes {
    private double textScore = 0.1;
    private final LoadImage loadImg = new LoadImage();
    private final Random random = new Random(0);

    public VisRes() {}

    public VisRes(double textScore) {
        this.textScore = textScore;
    }

    public Mat run(Object imgContent, List<RecResult> recRes, String fontPath) {
        List<Point[]> dtBoxes = recRes.stream().map(RecResult::getDtBoxes).collect(Collectors.toList());
        List<String> txts = recRes.stream().map(RecResult::getText).collect(Collectors.toList());
        List<Float> scores = recRes.stream().map(RecResult::getConfidence).collect(Collectors.toList());
        try {
            return this.run(imgContent, dtBoxes, txts, scores, fontPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public Mat runWord(Object imgContent, List<RecResult> recRes, String fontPath) {
        List<Point[]> dtBoxes = new ArrayList<>();
        List<String> txts = new ArrayList<>();
        List<Float> scores = new ArrayList<>();
        for (RecResult recRe : recRes) {
            WordBoxResult wordBoxResult = recRe.getWordBoxResult();
            dtBoxes.addAll(wordBoxResult.getSortedWordBoxList());
            txts.addAll(wordBoxResult.getWordBoxContentList());
            scores.addAll(wordBoxResult.getConfList());
        }
        try {
            return this.run(imgContent, dtBoxes, txts, scores, fontPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public Mat run(Object imgContent, List<Point[]> dtBoxes, List<String> txts,
                    List<Float> scores, String fontPath) throws IOException {
        if (txts == null || txts.isEmpty()) {
            return drawDtBoxes(imgContent, dtBoxes);
        }
        try {
            return drawOcrBoxTxt(imgContent, dtBoxes, txts, scores, fontPath, false);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }



    /**
     * 绘制检测框
     */
    public Mat drawDtBoxes(Object imgContent, List<Point[]> dtBoxes) {
        Mat img = null;
        try {
            img = loadImg.call(imgContent);
        } catch (LoadImageError e) {
            throw new RuntimeException(e);
        }

        for (int i = 0; i < dtBoxes.size(); i++) {
            Scalar color = getRandomColor();
            Point[] box = dtBoxes.get(i);
            MatOfPoint matBox = new MatOfPoint(box);
            Imgproc.polylines(img, Collections.singletonList(matBox), true, color, 1);
            matBox.release();

            Point startPoint = box[0];
            Imgproc.putText(img, String.valueOf(i), startPoint,
                    Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
        }
        return img;
    }

    /**
     * 绘制OCR框与文字
     */
    public Mat drawOcrBoxTxt(Object imgContent, List<Point[]> dtBoxes,
                             List<String> txts, List<Float> scores,
                             String fontPath, boolean fillLeftBox) throws Exception {
        fontPath = getFontPath(fontPath);
        Mat imgMat = loadImg.call(imgContent);

        BufferedImage image = matToBufferedImage(imgMat);
        imgMat.release();
        int w = image.getWidth();
        int h = image.getHeight();

        BufferedImage imgLeft = deepCopy(image);
        BufferedImage imgRight = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D gLeft = imgLeft.createGraphics();
        Graphics2D gRight = imgRight.createGraphics();
        gRight.setColor(Color.WHITE);
        gRight.fillRect(0, 0, w, h);

        try {
            // 加载字体文件
            Font baseFont = Font.createFont(Font.TRUETYPE_FONT, new File(fontPath));

            for (int i = 0; i < dtBoxes.size(); i++) {
                if (scores != null && scores.get(i) < textScore) continue;

                Scalar color = getRandomColor();
                java.awt.Color awtColor = new java.awt.Color(
                        (int) color.val[2], (int) color.val[1], (int) color.val[0]
                );

                Point[] pts = dtBoxes.get(i);
                Polygon polygon = new Polygon();
                for (Point p : pts) polygon.addPoint((int) p.x, (int) p.y);

                // 左图绘制框
                gLeft.setColor(awtColor);
                if (fillLeftBox) {
                    gLeft.fillPolygon(polygon);
                } else {
                    gLeft.setStroke(new BasicStroke(2));
                    gLeft.drawPolygon(polygon);
                }

                // 右图绘制框
                gRight.setColor(awtColor);
                gRight.drawPolygon(polygon);

                // 动态计算字体大小
                double boxHeight = getBoxHeight(pts);
                double boxWidth = getBoxWidth(pts);
                int fontSize = Math.max((int) (Math.min(boxHeight, boxWidth) * 0.8), 12);
                Font font = baseFont.deriveFont(Font.PLAIN, fontSize);
                gRight.setFont(font);
                gRight.setColor(Color.BLACK);
                FontMetrics fm = gRight.getFontMetrics(font);

                String txt = txts.get(i);

                // 根据框形状绘制文字
                if (boxHeight > 2 * boxWidth) {
                    // 竖排
                    int curY = (int) pts[0].y + fm.getAscent();
                    for (char c : txt.toCharArray()) {
                        gRight.drawString(String.valueOf(c), (int) pts[0].x + 3, curY);
                        curY += fm.getHeight();
                    }
                } else {
                    // 横排
                    gRight.drawString(txt, (int) pts[0].x, (int) (pts[0].y + fm.getAscent()));
                }
            }
        } finally {
            gLeft.dispose();
            gRight.dispose();
        }

        BufferedImage blended = blendImages(image, imgLeft, 0.5);
        BufferedImage merged = new BufferedImage(w * 2, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D gMerged = merged.createGraphics();
        try {
            gMerged.drawImage(blended, 0, 0, null);
            gMerged.drawImage(imgRight, w, 0, null);
        } finally {
            gMerged.dispose();
        }

        return bufferedImageToMat(merged);
    }


    private static String getFontPath(String fontPath) {
        if (fontPath == null || !new File(fontPath).exists()) {
            throw new RuntimeException("Font file not found: " + fontPath +
                    "\nDownload from: https://drive.google.com/file/d/1evWVX38EFNwTq_n5gTFgnlv8tdaNcyIA/view?usp=sharing");
        }
        return fontPath;
    }

    private Scalar getRandomColor() {
        return new Scalar(random.nextInt(256), random.nextInt(256), random.nextInt(256));
    }

    private static double getBoxHeight(Point[] box) {
        return Math.sqrt(Math.pow(box[0].x - box[3].x, 2) + Math.pow(box[0].y - box[3].y, 2));
    }

    private static double getBoxWidth(Point[] box) {
        return Math.sqrt(Math.pow(box[0].x - box[1].x, 2) + Math.pow(box[0].y - box[1].y, 2));
    }

    // ---------- 工具函数 ----------
    private static BufferedImage matToBufferedImage(Mat matrix) {
        int type = matrix.channels() == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_3BYTE_BGR;
        BufferedImage img = new BufferedImage(matrix.cols(), matrix.rows(), type);
        matrix.get(0, 0, ((java.awt.image.DataBufferByte) img.getRaster().getDataBuffer()).getData());
        return img;
    }

    /**
     * 将 BufferedImage 转为 Mat
     * @param bi 传入的 BufferedImage
     * @return 转换后的 Mat
     */
    private static Mat bufferedImageToMat(BufferedImage bi) {
        // 先转换为 TYPE_3BYTE_BGR 类型（OpenCV 默认是 BGR）
        if (bi.getType() != BufferedImage.TYPE_3BYTE_BGR) {
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

    private static BufferedImage deepCopy(BufferedImage bi) {
        BufferedImage copy = new BufferedImage(bi.getWidth(), bi.getHeight(), bi.getType());
        Graphics2D g = copy.createGraphics();
        g.drawImage(bi, 0, 0, null);
        g.dispose();
        return copy;
    }

    private static BufferedImage blendImages(BufferedImage img1, BufferedImage img2, double alpha) {
        BufferedImage blended = new BufferedImage(img1.getWidth(), img1.getHeight(), BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < img1.getHeight(); y++) {
            for (int x = 0; x < img1.getWidth(); x++) {
                int rgb1 = img1.getRGB(x, y);
                int rgb2 = img2.getRGB(x, y);
                int r = (int) (alpha * ((rgb2 >> 16) & 0xFF) + (1 - alpha) * ((rgb1 >> 16) & 0xFF));
                int g = (int) (alpha * ((rgb2 >> 8) & 0xFF) + (1 - alpha) * ((rgb1 >> 8) & 0xFF));
                int b = (int) (alpha * (rgb2 & 0xFF) + (1 - alpha) * (rgb1 & 0xFF));
                blended.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        return blended;
    }
}
