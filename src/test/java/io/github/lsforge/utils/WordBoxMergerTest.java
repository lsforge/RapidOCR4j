package io.github.lsforge.utils;

import io.github.lsforge.entity.WordBoxResult;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * WordBoxMerger 英文单词合并单元测试。
 */
public class WordBoxMergerTest {

    @Test
    void shouldMergeConsecutiveEnglishLettersIntoWords() {
        WordBoxResult input = createWordBoxResult(
                Arrays.asList("H", "e", "l", "l", "o", " ", "W", "o", "r", "l", "d"),
                Arrays.asList(0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.5f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f)
        );

        WordBoxResult merged = WordBoxMerger.mergeEnglishWords(input);

        List<String> contents = merged.getWordBoxContentList();
        Assertions.assertEquals(2, contents.size(), "应合并为两个单词");
        Assertions.assertEquals("Hello", contents.get(0));
        Assertions.assertEquals("World", contents.get(1));

        List<Point[]> boxes = merged.getSortedWordBoxList();
        Assertions.assertEquals(2, boxes.size());
        // 合并后的框应覆盖原单词所有字母
        Assertions.assertTrue(boxes.get(0)[0].x <= boxes.get(0)[1].x);
        Assertions.assertTrue(boxes.get(1)[0].x <= boxes.get(1)[1].x);
    }

    @Test
    void shouldPassThroughNonEnglishContent() {
        WordBoxResult input = createWordBoxResult(
                Arrays.asList("你", "好", " ", "世", "界"),
                Arrays.asList(0.9f, 0.9f, 0.5f, 0.9f, 0.9f)
        );

        WordBoxResult merged = WordBoxMerger.mergeEnglishWords(input);

        Assertions.assertEquals(Arrays.asList("你", "好", "世", "界"), merged.getWordBoxContentList());
    }

    @Test
    void shouldIgnoreWhitespace() {
        WordBoxResult input = createWordBoxResult(
                Arrays.asList("A", " ", "B"),
                Arrays.asList(0.9f, 0.5f, 0.9f)
        );

        WordBoxResult merged = WordBoxMerger.mergeEnglishWords(input);

        Assertions.assertEquals(Arrays.asList("A", "B"), merged.getWordBoxContentList());
    }

    @Test
    void shouldHandleNullLists() {
        WordBoxResult input = new WordBoxResult(null, null, null);
        WordBoxResult merged = WordBoxMerger.mergeEnglishWords(input);
        Assertions.assertSame(input, merged, "空列表应原样返回");
    }

    @Test
    void shouldHandleEmptyLists() {
        WordBoxResult input = new WordBoxResult(
                Collections.emptyList(),
                Collections.emptyList(),
                Collections.emptyList()
        );
        WordBoxResult merged = WordBoxMerger.mergeEnglishWords(input);
        Assertions.assertTrue(merged.getWordBoxContentList().isEmpty());
    }

    private static WordBoxResult createWordBoxResult(List<String> contents, List<Float> confs) {
        List<Point[]> boxes = new ArrayList<>();
        double x = 0;
        for (int i = 0; i < contents.size(); i++) {
            boxes.add(new Point[]{
                    new Point(x, 0),
                    new Point(x + 10, 0),
                    new Point(x + 10, 10),
                    new Point(x, 10)
            });
            x += 10;
        }
        return new WordBoxResult(contents, boxes, confs);
    }
}
