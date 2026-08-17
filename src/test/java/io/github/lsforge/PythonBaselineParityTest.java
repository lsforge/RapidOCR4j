package io.github.lsforge;

import io.github.lsforge.entity.OcrResult;
import io.github.lsforge.entity.RecResult;
import io.github.lsforge.utils.OpencvLoader;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 与 Python 端 RapidOCR（默认配置）对 ch_en_num.jpg 的输出做精度比对。
 *
 * <p>基线数据来自 Python 端 RapidOCROutput：18 行文本、置信度、检测框四点坐标（已映射回原图坐标系）。
 * 生成环境：RapidOCR-main（v3.9.x，D:\project\RapidOCR-main）+ PP-OCRv6 det/rec small 模型
 * （MD5 与本工程 src/main/resources/models 一致），2026-08-17 重新生成并经逐行对拍验证。</p>
 */
public class PythonBaselineParityTest {

    private static final String TEST_IMG = "src/test/resources/ch_en_num.jpg";

    private static final double BOX_EPS = 2.0;
    private static final float SCORE_EPS = 1e-3f;

    private static RapidOCR sharedEngine;

    /**
     * 单行基线数据：检测框（4 顶点 tl/tr/br/bl）、识别文本、置信度。
     */
    private static final class BaselineRow {
        private final float[][] box; // 4 x 2
        private final String text;
        private final float score;

        BaselineRow(float[][] box, String text, float score) {
            this.box = box;
            this.text = text;
            this.score = score;
        }
    }

    /**
     * Python 端 RapidOCR 对 ch_en_num.jpg 的完整输出基线。
     */
    private static final List<BaselineRow> PYTHON_BASELINE = Arrays.asList(
            new BaselineRow(new float[][]{{0f, 0f}, {322f, 0f}, {322f, 108f}, {0f, 101f}},
                    "正品促销", 0.99987f),
            new BaselineRow(new float[][]{{67f, 93f}, {256f, 93f}, {256f, 128f}, {67f, 128f}},
                    "大桶装更划算", 0.99981f),
            new BaselineRow(new float[][]{{64f, 140f}, {259f, 140f}, {259f, 167f}, {64f, 167f}},
                    "强力去污符合国标", 0.99977f),
            new BaselineRow(new float[][]{{104f, 167f}, {219f, 167f}, {219f, 183f}, {104f, 183f}},
                    "-40℃深度防冻不结冰", 0.97438f),
            new BaselineRow(new float[][]{{32f, 232f}, {82f, 233f}, {81f, 255f}, {31f, 253f}},
                    "10.0起", 0.99489f),
            new BaselineRow(new float[][]{{33f, 225f}, {63f, 225f}, {63f, 236f}, {33f, 236f}},
                    "日常价￥", 0.88060f),
            new BaselineRow(new float[][]{{136f, 220f}, {190f, 220f}, {190f, 251f}, {136f, 251f}},
                    "直击", 0.99168f),
            new BaselineRow(new float[][]{{254f, 232f}, {304f, 232f}, {304f, 254f}, {254f, 254f}},
                    "10.0起", 0.99985f),
            new BaselineRow(new float[][]{{258f, 225f}, {288f, 225f}, {288f, 236f}, {258f, 236f}},
                    "日常价¥", 0.88897f),
            new BaselineRow(new float[][]{{136f, 242f}, {190f, 242f}, {190f, 273f}, {136f, 273f}},
                    "底价", 0.99912f),
            new BaselineRow(new float[][]{{124f, 285f}, {210f, 285f}, {210f, 340f}, {124f, 340f}},
                    "5.8", 0.99999f),
            new BaselineRow(new float[][]{{97f, 318f}, {130f, 318f}, {130f, 330f}, {97f, 330f}},
                    "券后价¥", 0.87586f),
            new BaselineRow(new float[][]{{203f, 314f}, {225f, 314f}, {225f, 335f}, {203f, 335f}},
                    "起", 1.00000f),
            new BaselineRow(new float[][]{{111f, 340f}, {212f, 340f}, {212f, 356f}, {111f, 356f}},
                    "惊喜福利不容错过", 0.99995f),
            new BaselineRow(new float[][]{{37f, 358f}, {155f, 358f}, {155f, 386f}, {37f, 386f}},
                    "极速发货", 0.99998f),
            new BaselineRow(new float[][]{{173f, 359f}, {289f, 359f}, {289f, 385f}, {173f, 385f}},
                    "冰点标准", 0.99990f),
            new BaselineRow(new float[][]{{63f, 388f}, {154f, 388f}, {154f, 414f}, {63f, 414f}},
                    "破损就赔", 0.99903f),
            new BaselineRow(new float[][]{{173f, 387f}, {289f, 387f}, {289f, 415f}, {173f, 415f}},
                    "假一赔十", 0.99746f)
    );

    @BeforeAll
    static void init() {
        OpencvLoader.loadOpencvLib();
        sharedEngine = RapidOCR.create();
    }

    @Test
    void shouldMatchPythonBaseline() throws Exception {
        OcrResult result = sharedEngine.run(TEST_IMG);
        assertPythonMatch(result);
    }

    private static void assertPythonMatch(OcrResult result) {
        Assertions.assertNotNull(result, "结果不应为 null");
        List<RecResult> actual = result.getRecRes();
        Assertions.assertEquals(PYTHON_BASELINE.size(), actual.size(),
                "识别行数应与 Python 基线一致");

        Map<String, List<Integer>> expectedIndexByText = new HashMap<>();
        for (int i = 0; i < PYTHON_BASELINE.size(); i++) {
            String key = PYTHON_BASELINE.get(i).text;
            expectedIndexByText.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
        }
        boolean[] used = new boolean[PYTHON_BASELINE.size()];

        for (int i = 0; i < actual.size(); i++) {
            RecResult actualRow = actual.get(i);
            String text = actualRow.getText();
            String ctx = "第 " + i + " 行（文本 '" + text + "'）";
            Assertions.assertNotNull(text, ctx + ": 文本为 null");
            Assertions.assertNull(actualRow.getWordBoxResult(),
                    ctx + ": 默认配置不应返回 wordBoxResult");

            List<Integer> candidates = expectedIndexByText.get(text.trim());
            Assertions.assertNotNull(candidates, ctx + ": 在 Python 基线中找不到对应文本");

            int matchedIndex = pickByNearestX(actualRow.getDtBoxes(), candidates, ctx);
            Assertions.assertFalse(used[matchedIndex],
                    ctx + ": Python 基线第 " + matchedIndex + " 行已被匹配");
            used[matchedIndex] = true;

            BaselineRow expected = PYTHON_BASELINE.get(matchedIndex);
            assertScoreClose(expected.score, actualRow.getConfidence(), ctx);
            assertBoxesClose(expected.box, actualRow.getDtBoxes(), ctx);
        }

        for (int i = 0; i < used.length; i++) {
            Assertions.assertTrue(used[i],
                    "Python 基线第 " + i + " 行（'" + PYTHON_BASELINE.get(i).text + "'）未被匹配");
        }
    }

    /**
     * 对重复文本候选，按检测框第一个顶点的 x 坐标最近原则挑选。
     */
    private static int pickByNearestX(Point[] actualBox, List<Integer> candidates, String ctx) {
        Assertions.assertNotNull(actualBox, ctx + ": 检测框为 null，无法匹配重复文本");
        Assertions.assertTrue(actualBox.length > 0, ctx + ": 检测框为空");
        double actualX = actualBox[0].x;

        int bestIndex = -1;
        double bestDist = Double.MAX_VALUE;
        for (int idx : candidates) {
            float expectedX = PYTHON_BASELINE.get(idx).box[0][0];
            double dist = Math.abs(expectedX - actualX);
            if (dist < bestDist) {
                bestDist = dist;
                bestIndex = idx;
            }
        }
        Assertions.assertTrue(bestIndex >= 0, ctx + ": 未找到匹配的候选行");
        return bestIndex;
    }

    private static void assertScoreClose(float expected, float actual, String context) {
        Assertions.assertEquals(expected, actual, SCORE_EPS, context + ": 置信度超差");
    }

    private static void assertBoxesClose(float[][] expected, Point[] actual, String context) {
        Assertions.assertNotNull(actual, context + ": 检测框为 null");
        Assertions.assertEquals(4, actual.length, context + ": 检测框顶点数不为 4");

        for (int v = 0; v < 4; v++) {
            Assertions.assertEquals(expected[v][0], actual[v].x, BOX_EPS,
                    context + ": 顶点 " + v + " 的 x 坐标超差");
            Assertions.assertEquals(expected[v][1], actual[v].y, BOX_EPS,
                    context + ": 顶点 " + v + " 的 y 坐标超差");
        }
    }
}
