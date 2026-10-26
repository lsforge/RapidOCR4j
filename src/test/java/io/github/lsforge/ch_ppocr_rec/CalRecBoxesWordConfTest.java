package io.github.lsforge.ch_ppocr_rec;

import io.github.lsforge.cal_rec_boxes.CalRecBoxes;
import io.github.lsforge.entity.TupleResult;
import io.github.lsforge.entity.WordBoxResult;
import io.github.lsforge.utils.OpencvLoader;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * 对齐 Python b40c0f8：word 分数取自身字符均值，空格置信度不混入。
 * 复刻 Python test_word_conf_from_own_chars 的合成数据（空格 0.99，字符 0.9/0.6/0.8/0.5）
 */
public class CalRecBoxesWordConfTest {

    /** 字典条目数：blank + a b c d 中 文 + 空格 */
    private static final int NUM_CLASSES = 8;

    @BeforeAll
    static void setUp() {
        OpencvLoader.loadOpencvLib();
    }

    /**
     * 构造 (1, 2T, C) 概率矩阵：偶数时间步为字符（CTC 展开），奇数步为 blank；
     * 空格字符置信度固定 0.99，其余按 charProbs 顺序取值（与 Python 测试一致）。
     */
    private static float[][][] buildPreds(String chars, List<Float> charProbs) {
        List<String> character = Arrays.asList("a", "b", "c", "d", "中", "文");
        float[][][] preds = new float[1][2 * chars.length()][NUM_CLASSES];
        int probCursor = 0;
        for (int i = 0; i < chars.length(); i++) {
            char c = chars.charAt(i);
            int charIdx;
            float prob;
            if (c == ' ') {
                charIdx = NUM_CLASSES - 1; // 空格条目在字典末尾
                prob = 0.99f;
            } else {
                charIdx = character.indexOf(String.valueOf(c)) + 1; // blank 占 0 号
                prob = charProbs.get(probCursor++);
            }
            preds[0][2 * i][charIdx] = prob;
            preds[0][2 * i + 1][0] = 0.95f;
        }
        return preds;
    }

    private static WordBoxResult decodeAndCalBoxes(String chars, List<Float> charProbs, boolean returnWordLevel) {
        CTCLabelDecode decoder = new CTCLabelDecode(Arrays.asList("a", "b", "c", "d", "中", "文"), null);
        List<DecodeResult> decodeResults = decoder.call(
                buildPreds(chars, charProbs), true, Collections.singletonList(1f), 1f);
        DecodeResult r = decodeResults.get(0);
        assertEquals(chars, r.getText(), "解码文本应还原输入字符序列");

        Mat img = new Mat(48, 320, CvType.CV_8UC3);
        Point[] box = {
                new Point(0, 0), new Point(320, 0), new Point(320, 48), new Point(0, 48)
        };
        List<TupleResult> out = new CalRecBoxes().call(
                Collections.singletonList(img),
                Collections.singletonList(box),
                Collections.singletonList(new TupleResult(r.getText(), r.getConfidence(), r.getWordBoxInfo())),
                returnWordLevel);
        return out.get(0).getWordBoxResult();
    }

    @Test
    void wordLevelConfShouldBeMeanOfOwnChars() {
        WordBoxResult result = decodeAndCalBoxes("ab cd",
                Arrays.asList(0.9f, 0.6f, 0.8f, 0.5f), true);

        assertEquals(Arrays.asList("ab", "cd"), result.getWordBoxContentList());
        // ab -> mean(0.9, 0.6) = 0.75；cd -> mean(0.8, 0.5) = 0.65；空格的 0.99 不得混入
        assertEquals(0.75f, result.getConfList().get(0), 1e-4);
        assertEquals(0.65f, result.getConfList().get(1), 1e-4);
    }

    @Test
    void singleCharLevelConfShouldFollowOwnChars() {
        WordBoxResult result = decodeAndCalBoxes("ab cd",
                Arrays.asList(0.9f, 0.6f, 0.8f, 0.5f), false);

        assertEquals(Arrays.asList("a", "b", "c", "d"), result.getWordBoxContentList());
        // 旧实现返回整行 confList（含空格 0.99），数量与内容错位；新行为逐字符对齐
        assertEquals(4, result.getConfList().size());
        assertEquals(Arrays.asList(0.9f, 0.6f, 0.8f, 0.5f), result.getConfList());
    }

    @Test
    void chineseLineShouldKeepOwnConf() {
        WordBoxResult result = decodeAndCalBoxes("中 文",
                new ArrayList<>(Arrays.asList(0.9f, 0.6f)), false);

        assertEquals(Arrays.asList("中", "文"), result.getWordBoxContentList());
        assertEquals(Arrays.asList(0.9f, 0.6f), result.getConfList());
    }
}
