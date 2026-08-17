package io.github.lsforge.utils;

import io.github.lsforge.entity.WordBoxResult;
import org.opencv.core.Point;
import java.util.*;

public class WordBoxMerger {

    /**
     * 英文字母合并为单词
     */
    public static WordBoxResult mergeEnglishWords(WordBoxResult wordBoxResult) {
        List<String> contents = wordBoxResult.getWordBoxContentList();
        List<Point[]> boxes = wordBoxResult.getSortedWordBoxList();
        List<Float> confs = wordBoxResult.getConfList();

        if (contents == null || boxes == null || confs == null) {
            return wordBoxResult;
        }

        List<String> newContents = new ArrayList<>();
        List<Point[]> newBoxes = new ArrayList<>();
        List<Float> newConfs = new ArrayList<>();

        StringBuilder currentWord = new StringBuilder();
        List<Point[]> currentBoxes = new ArrayList<>();
        List<Float> currentConfs = new ArrayList<>();
        boolean merging = false;

        for (int i = 0; i < contents.size(); i++) {
            String content = contents.get(i);
            Point[] box = boxes.get(i);
            Float conf = confs.get(i);

            boolean isLetter = content != null && content.matches("^[A-Za-z]+$");

            if (isLetter) {
                // 字母，加入当前单词
                currentWord.append(content);
                currentBoxes.add(box);
                currentConfs.add(conf);
                merging = true;
            } else {
                // 非字母或空格，结束当前单词
                if (merging && currentWord.length() > 0) {
                    newContents.add(currentWord.toString());
                    newBoxes.add(mergePoints(currentBoxes));
                    newConfs.add(average(currentConfs));
                    currentWord.setLength(0);
                    currentBoxes.clear();
                    currentConfs.clear();
                    merging = false;
                }

                // 空格可以忽略，如果不是空格再添加到结果
                if (content != null && !content.trim().isEmpty()) {
                    newContents.add(content);
                    newBoxes.add(box);
                    newConfs.add(conf);
                }
            }
        }

        // 处理最后一组未关闭的单词
        if (merging && currentWord.length() > 0) {
            newContents.add(currentWord.toString());
            newBoxes.add(mergePoints(currentBoxes));
            newConfs.add(average(currentConfs));
        }

        return new WordBoxResult(newContents, newBoxes, newConfs);
    }

    /**
     * 合并多个 bounding box 点为一个最小外接矩形
     */
    private static Point[] mergePoints(List<Point[]> boxes) {
        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE;
        double maxX = Double.MIN_VALUE, maxY = Double.MIN_VALUE;

        for (Point[] box : boxes) {
            if (box == null) continue;
            for (Point p : box) {
                if (p == null) continue;
                minX = Math.min(minX, p.x);
                minY = Math.min(minY, p.y);
                maxX = Math.max(maxX, p.x);
                maxY = Math.max(maxY, p.y);
            }
        }

        return new Point[]{
                new Point(minX, minY),
                new Point(maxX, minY),
                new Point(maxX, maxY),
                new Point(minX, maxY)
        };
    }

    /**
     * 求置信度平均值
     */
    private static Float average(List<Float> values) {
        if (values == null || values.isEmpty()) return 0f;
        float sum = 0f;
        for (Float v : values) sum += v;
        return sum / values.size();
    }
}
