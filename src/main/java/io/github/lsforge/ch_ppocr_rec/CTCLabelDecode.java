package io.github.lsforge.ch_ppocr_rec;

import io.github.lsforge.entity.WordBoxInfo;
import lombok.Data;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * 用于存储 decode 后的结果
 */
@Data
class DecodeResult {
    private String text;        // 解码出的文本
    private float confidence;   // 平均置信度

    // 存储文本坐标/word box 等信息
    private WordBoxInfo wordBoxInfo;

    public DecodeResult(String text, float confidence) {
        this.text = text;
        this.confidence = confidence;
    }

    public DecodeResult(String text, float confidence, WordBoxInfo wordBoxInfo) {
        this.text = text;
        this.confidence = confidence;
        this.wordBoxInfo = wordBoxInfo;
    }
}

/**
 * CTC 解码
 */
public class CTCLabelDecode {
    // 存储字符集
    private List<String> character;
    // 存储字符到索引的映射
    private Map<String, Integer> dict;

    /**
     * 构造方法
     *
     * @param characterList 直接传入的字符列表（若非空，可以优先使用）
     * @param characterPath 字符文件路径（若 characterList 不存在，则从文件加载）
     */
    public CTCLabelDecode(List<String> characterList, String characterPath) {
        this.character = getCharacter(characterList, characterPath);
        this.dict = new HashMap<>();
        for (int i = 0; i < this.character.size(); i++) {
            this.dict.put(this.character.get(i), i);
        }
    }

    /**
     * 解码入口方法
     *
     * @param preds         预测输出 (形状为 batchSize x timeSteps x numClasses)
     * @param returnWordBox 是否返回字符分组、坐标等附加信息
     * @param whRatioList   与图像宽高比相关的列表
     * @param maxWhRatio    全局最大宽高比
     * @return 解码结果列表
     */
    public List<DecodeResult> call(float[][][] preds, boolean returnWordBox, List<Float> whRatioList, float maxWhRatio) {
        // 1. 计算 predsIdx (即 argmax) 和 predsProb (即 max 值)
        //    preds 的形状: batchSize x timeSteps x numClasses
        //    predsIdx 的形状: batchSize x timeSteps
        //    predsProb 的形状: batchSize x timeSteps
        int batchSize = preds.length;
        int timeSteps = preds[0].length;
        int numClasses = preds[0][0].length;

        int[][] predsIdx = new int[batchSize][timeSteps];
        float[][] predsProb = new float[batchSize][timeSteps];

        // 循环计算 argmax 和 max（v6 字典类数上万，此循环是 rec 后处理热点；
        //   行引用外提减少每元素的多级寻址，初始 -1/-Inf 语义保持不变）
        for (int b = 0; b < batchSize; b++) {
            float[][] plane = preds[b];
            for (int t = 0; t < timeSteps; t++) {
                float[] row = plane[t];
                float maxVal = Float.NEGATIVE_INFINITY;
                int maxIndex = -1;
                for (int c = 0; c < numClasses; c++) {
                    float val = row[c];
                    if (val > maxVal) {
                        maxVal = val;
                        maxIndex = c;
                    }
                }
                predsIdx[b][t] = maxIndex;
                predsProb[b][t] = maxVal;
            }
        }

        // 2. 进行文本解码
        List<DecodeResult> decodeResults = this.decode(predsIdx, predsProb, returnWordBox, true);

        // 3. 若需要对 wordBox 做进一步处理
        if (returnWordBox) {
            for (int i = 0; i < decodeResults.size(); i++) {
                // 对 decodeResults 里某些字段做修正，调整 decodeResults 内的数据
                float whRatio = whRatioList.get(i);
                decodeResults.get(i).getWordBoxInfo().setTextIndexLen(decodeResults.get(i).getWordBoxInfo().getTextIndexLen() * (whRatio / maxWhRatio));
            }
        }

        return decodeResults;
    }

    /**
     * 获取字符集
     */
    private List<String> getCharacter(List<String> characterList, String characterPath) {
        // 如果外部没有直接传入 characterList，则从文件中读取
        if ((characterList == null || characterList.isEmpty()) && characterPath == null) {
            throw new IllegalArgumentException("character must not be null");
        }

        List<String> finalList = null;
        if (characterList != null && !characterList.isEmpty()) {
            finalList = new ArrayList<>(characterList);
        }

        // 如果外部没有传入，则尝试从文件加载
        if (finalList == null && characterPath != null) {
            finalList = readCharacterFile(characterPath);
        }

        if (finalList == null) {
            throw new IllegalArgumentException("character must not be null");
        }

        // 插入空格
        finalList = insertSpecialChar(finalList, " ", finalList.size());
        // 插入 blank
        finalList = insertSpecialChar(finalList, "blank", 0);
        return finalList;
    }

    /**
     * 从文件中读取字符列表
     */
    private List<String> readCharacterFile(String characterPath) {
        List<String> characterList = new ArrayList<>();
        InputStream inputStream = null;
        try {
            Path path = Paths.get(characterPath);
            if (path.isAbsolute()) {
                inputStream = Files.newInputStream(path);
            } else {
                inputStream = Thread.currentThread().getContextClassLoader().getResourceAsStream(characterPath);
                if (inputStream == null) {
                    throw new FileNotFoundException("Resource not found: " + characterPath);
                }
            }
            try (BufferedReader br = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
                String line;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (!line.isEmpty()) {
                        characterList.add(line);
                    }
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("读取字符文件失败: " + characterPath);
        }
        return characterList;
    }

    /**
     * 在指定位置插入特殊字符
     */
    private List<String> insertSpecialChar(List<String> characterList, String specialChar, int loc) {
        if (loc < 0 || loc > characterList.size()) {
            loc = characterList.size();
        }
        characterList.add(loc, specialChar);
        return characterList;
    }

    /**
     * 解码核心逻辑
     *
     * @param textIndex         预测得到的字符索引 (batchSize x timeSteps)
     * @param textProb          对应的最大概率 (batchSize x timeSteps)
     * @param returnWordBox     是否返回字分组、位置信息
     * @param isRemoveDuplicate 是否移除连续相同的字符 (CTC 的常用操作)
     * @return 解码结果
     */
    private List<DecodeResult> decode(int[][] textIndex, float[][] textProb, boolean returnWordBox, boolean isRemoveDuplicate) {
        List<DecodeResult> resultList = new ArrayList<>();

        int batchSize = textIndex.length;

        for (int b = 0; b < batchSize; b++) {
            int[] curTextIndex = textIndex[b];
            float[] curTextProb = textProb[b];

            // 移除连续重复字符 + 忽略 blank（index 恒为 0，直接比较免装箱与 List 查询）。
            // 两个判断都是置 selection[i]=false，合单趟遍历结果不变
            boolean[] selection = new boolean[curTextIndex.length];
            Arrays.fill(selection, true);

            if (isRemoveDuplicate && curTextIndex.length >= 2) {
                for (int i = 1; i < curTextIndex.length; i++) {
                    if (curTextIndex[i] == curTextIndex[i - 1]) {
                        selection[i] = false;
                    }
                }
            }
            for (int i = 0; i < curTextIndex.length; i++) {
                if (curTextIndex[i] == 0) {
                    selection[i] = false;
                }
            }

            // 根据 selection 获取保留的字符和概率
            // 注意：字典条目可能是多字符（如 v6 字典含 362 个多字符条目），必须整条目保留。
            // confList 仅在 returnWordBox 时构建（WordBoxInfo 需要），否则边扫边累积，免装箱
            List<String> charList = new ArrayList<>();
            List<Float> confList = returnWordBox ? new ArrayList<>() : null;
            float sum = 0.0f;
            int count = 0;
            for (int i = 0; i < curTextIndex.length; i++) {
                if (selection[i]) {
                    int textId = curTextIndex[i];
                    // 若越界（模型输出含训练占位类），回退为空格
                    if (textId >= 0 && textId < character.size()) {
                        charList.add(character.get(textId));
                    } else {
                        charList.add(" ");
                    }
                    sum += curTextProb[i];
                    count++;
                    if (confList != null) {
                        confList.add(curTextProb[i]);
                    }
                }
            }

            // 若没有选中的，则默认给置信度 0
            if (count == 0) {
                sum = 0.0f;
                count = 1;
                if (confList != null) {
                    confList.add(0.0f);
                }
            }

            // 组装最终字符串
            StringBuilder sb = new StringBuilder();
            for (String s : charList) {
                sb.append(s);
            }
            String decodedText = sb.toString();

            // 计算平均置信度（累加次序保持行主序，与旧实现位一致）
            float avgConf = sum / count;

            // 若不需要 wordBox，直接返回 (text, avgConf)
            if (!returnWordBox) {
                resultList.add(new DecodeResult(decodedText, avgConf));
            } else {
                // 需要获取分词信息（按字典条目遍历，条目数与 selection 选中位置一一对应）
                WordInfo wi = this.getWordInfo(charList, selection);

                // 组装 WordBoxInfo：分词分组 + 逐条目置信度，供 CalRecBoxes 计算字/词级坐标
                WordBoxInfo wordBoxInfo = new WordBoxInfo((double) textIndex[b].length, wi.wordList, wi.wordColList, wi.stateList, confList);
                resultList.add(new DecodeResult(decodedText, avgConf, wordBoxInfo));
            }
        }

        return resultList;
    }

    /**
     * 用于描述 getWordInfo 返回的结果
     */
    private static class WordInfo {
        List<List<String>> wordList;
        List<List<Integer>> wordColList;
        List<String> stateList;

        public WordInfo(
                List<List<String>> wordList,
                List<List<Integer>> wordColList,
                List<String> stateList
        ) {
            this.wordList = wordList;
            this.wordColList = wordColList;
            this.stateList = stateList;
        }
    }

    /**
     * 获取文字分组信息（按字典条目遍历，与 Python get_word_info 语义对齐）
     *
     * @param charList  解码保留的字典条目列表（条目可能是多字符），与 selection 选中位置一一对应
     * @param selection 与每个时间步对应的选中状态
     * @return 分组结果
     */
    private WordInfo getWordInfo(List<String> charList, boolean[] selection) {
        // 找到 selection 为 true 的下标（与 charList 条目一一对应）
        List<Integer> validCols = new ArrayList<>();
        for (int i = 0; i < selection.length; i++) {
            if (selection[i]) {
                validCols.add(i);
            }
        }

        // 计算相邻列间距
        List<Integer> colWidth = new ArrayList<>();
        for (int i = 0; i < validCols.size(); i++) {
            if (i == 0) {
                // 对齐 Python: col_width[0] = min(首条目是中文则 3 否则 2, 首列位置)
                colWidth.add(Math.min(isCjk(charList.get(0)) ? 3 : 2, validCols.get(0)));
            } else {
                colWidth.add(validCols.get(i) - validCols.get(i - 1));
            }
        }

        List<List<String>> wordList = new ArrayList<>();
        List<List<Integer>> wordColList = new ArrayList<>();
        List<String> stateList = new ArrayList<>();

        List<String> wordContent = new ArrayList<>();
        List<Integer> wordColContent = new ArrayList<>();
        String state = null;

        // 遍历每个字典条目，并根据中英文或间距进行分组
        for (int i = 0; i < charList.size(); i++) {
            String entry = charList.get(i);

            // 对齐 Python: 遇到空格条目立即断词并跳过该空格（不并入任何单词）
            if (entry.trim().isEmpty()) {
                if (!wordContent.isEmpty()) {
                    wordList.add(new ArrayList<>(wordContent));
                    wordColList.add(new ArrayList<>(wordColContent));
                    stateList.add(state);
                    wordContent.clear();
                    wordColContent.clear();
                }
                continue;
            }

            String cState = isCjk(entry) ? "cn" : "en&num";

            if (state == null) {
                state = cState;
            }

            // 如果前后字符状态不同 或者 间距大于 5，则认为需要断句
            if (!state.equals(cState) || (i < colWidth.size() && colWidth.get(i) > 5)) {
                if (!wordContent.isEmpty()) {
                    wordList.add(new ArrayList<>(wordContent));
                    wordColList.add(new ArrayList<>(wordColContent));
                    stateList.add(state);
                    wordContent.clear();
                    wordColContent.clear();
                }
                state = cState;
            }

            wordContent.add(entry);
            if (i < validCols.size()) {
                wordColContent.add(validCols.get(i));
            }
        }

        // 收尾
        if (!wordContent.isEmpty()) {
            wordList.add(wordContent);
            wordColList.add(wordColContent);
            stateList.add(state);
        }

        return new WordInfo(wordList, wordColList, stateList);
    }

    /**
     * 判断字典条目是否为中文字符（按条目首个字符的码位判断，沿用基本区范围）
     */
    private static boolean isCjk(String entry) {
        if (entry == null || entry.isEmpty()) {
            return false;
        }
        char ch = entry.charAt(0);
        return ch >= 0x4E00 && ch <= 0x9FA5;
    }

}
