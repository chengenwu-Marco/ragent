/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nageoffer.ai.ragent.rag.core.rewrite;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.nageoffer.ai.ragent.infra.util.LLMResponseCleaner;
import com.nageoffer.ai.ragent.rag.config.RAGConfigProperties;
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.framework.trace.RagTraceNode;
import com.nageoffer.ai.ragent.infra.chat.LLMService;
import com.nageoffer.ai.ragent.infra.enums.Tier;
import com.nageoffer.ai.ragent.rag.core.prompt.PromptTemplateLoader;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.QUERY_REWRITE_AND_SPLIT_PROMPT_PATH;

/**
 * 查询预处理：改写 + 拆分多问句
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class MultiQuestionRewriteService implements QueryRewriteService {

    /**
     * 疑问标记：命中说明输入是提问，不做闲聊守卫拦截
     */
    private static final Pattern QUESTION_MARKER_PATTERN = Pattern.compile("[?？吗呢么怎么如何什么为什么为啥哪多少几]");

    /**
     * 反馈/闲聊关键词：短输入命中且无疑问标记时，视为非问题类输入
     */
    private static final Pattern CHITCHAT_KEYWORD_PATTERN = Pattern.compile(
            "不错|好的|谢谢|多谢|辛苦|收到|明白|了解|知道|可以|挺好|很棒|很好|厉害|再见|拜拜|你好|[Oo][Kk]");

    /**
     * 请求类前缀：以这些词开头的输入是陈述式提问，不做闲聊守卫拦截
     */
    private static final Pattern REQUEST_PREFIX_PATTERN = Pattern.compile(
            "^(请|帮我|麻烦|想知道|想了解|请问|问下|讲讲|说说|介绍一下|怎么)");

    private final LLMService llmService;
    private final RAGConfigProperties ragConfigProperties;
    private final QueryTermMappingService queryTermMappingService;
    private final PromptTemplateLoader promptTemplateLoader;

    @Override
    @RagTraceNode(name = "query-rewrite", type = "REWRITE")
    public String rewrite(String userQuestion) {
        return rewriteAndSplit(userQuestion).rewrittenQuestion();
    }

    @Override
    public RewriteResult rewriteWithSplit(String userQuestion) {
        return rewriteAndSplit(userQuestion);
    }

    @Override
    @RagTraceNode(name = "query-rewrite-and-split", type = "REWRITE")
    public RewriteResult rewriteWithSplit(String userQuestion, List<ChatMessage> history) {
        String normalizedQuestion = queryTermMappingService.normalize(userQuestion);

        RewriteResult nonQuestionResult = skipIfNonQuestion(normalizedQuestion);
        if (nonQuestionResult != null) {
            return nonQuestionResult;
        }

        if (!ragConfigProperties.getQueryRewriteEnabled()) {
            List<String> subs = ruleBasedSplit(normalizedQuestion);
            return new RewriteResult(normalizedQuestion, subs);
        }

        return callLLMRewriteAndSplit(normalizedQuestion, userQuestion, history);
    }

    /**
     * 先用默认改写做归一化，再进行多问句拆分。
     */
    private RewriteResult rewriteAndSplit(String userQuestion) {
        String normalizedQuestion = queryTermMappingService.normalize(userQuestion);

        RewriteResult nonQuestionResult = skipIfNonQuestion(normalizedQuestion);
        if (nonQuestionResult != null) {
            return nonQuestionResult;
        }

        // 开关关闭：直接做规则归一化 + 规则拆分
        if (!ragConfigProperties.getQueryRewriteEnabled()) {
            List<String> subs = ruleBasedSplit(normalizedQuestion);
            return new RewriteResult(normalizedQuestion, subs);
        }

        return callLLMRewriteAndSplit(normalizedQuestion, userQuestion, List.of());
    }

    /**
     * 非问题类输入（反馈、评价、闲聊）原样返回，不调用 LLM 改写。
     * <p>
     * 否则 LLM 会结合会话历史把"回答的不错"这类反馈误改写成历史话题
     * （如"集团的发票情况"），导致后续检索命中无关知识库。
     */
    private RewriteResult skipIfNonQuestion(String normalizedQuestion) {
        if (!isNonQuestionInput(normalizedQuestion)) {
            return null;
        }
        log.info("检测到非问题类输入，跳过查询改写，原样返回：{}", normalizedQuestion);
        return new RewriteResult(normalizedQuestion, List.of(normalizedQuestion));
    }

    /**
     * 短输入、无疑问标记且命中反馈/闲聊关键词时判定为非问题类输入。
     * <p>
     * 长度上限内保守匹配：即使误拦截陈述式提问，也只是跳过改写，
     * 原始问题仍会进入意图识别与检索，不会导致答非所问。
     */
    private boolean isNonQuestionInput(String normalizedQuestion) {
        if (StrUtil.isBlank(normalizedQuestion)) {
            return true;
        }
        if (normalizedQuestion.length() > 20) {
            return false;
        }
        if (REQUEST_PREFIX_PATTERN.matcher(normalizedQuestion).find()) {
            return false;
        }
        if (QUESTION_MARKER_PATTERN.matcher(normalizedQuestion).find()) {
            return false;
        }
        return CHITCHAT_KEYWORD_PATTERN.matcher(normalizedQuestion).find();
    }

    private RewriteResult callLLMRewriteAndSplit(String normalizedQuestion,
                                                 String originalQuestion,
                                                 List<ChatMessage> history) {
        String systemPrompt = promptTemplateLoader.load(QUERY_REWRITE_AND_SPLIT_PROMPT_PATH);
        ChatRequest req = buildRewriteRequest(systemPrompt, normalizedQuestion, history);

        // 快速档调用；解析失败或调用失败均用归一化问题兜底（档位内多候选已提供传输容错，不再跨档升级）
        RewriteResult fallback = new RewriteResult(normalizedQuestion, List.of(normalizedQuestion));
        RewriteResult result;
        try {
            RewriteResult parsed = parseRewriteAndSplit(llmService.chat(req, Tier.FAST));
            result = parsed != null ? parsed : fallback;
        } catch (Exception e) {
            log.warn("查询改写 LLM 调用失败，使用归一化问题兜底", e);
            result = fallback;
        }

        log.info("""
                RAG用户问题查询改写+拆分：
                原始问题：{}
                归一化后：{}
                改写结果：{}
                子问题：{}
                """, originalQuestion, normalizedQuestion, result.rewrittenQuestion(), result.subQuestions());
        return result;
    }

    private ChatRequest buildRewriteRequest(String systemPrompt,
                                            String question,
                                            List<ChatMessage> history) {
        List<ChatMessage> messages = new ArrayList<>();
        if (StrUtil.isNotBlank(systemPrompt)) {
            messages.add(ChatMessage.system(systemPrompt));
        }

        // 只保留最近 1-2 轮的 User 和 Assistant 消息
        // 过滤掉 System 摘要，避免 Token 浪费
        if (CollUtil.isNotEmpty(history)) {
            List<ChatMessage> recentHistory = history.stream()
                    .filter(msg -> msg.getRole() == ChatMessage.Role.USER
                            || msg.getRole() == ChatMessage.Role.ASSISTANT)
                    .skip(Math.max(0, history.size() - 4))  // 最多保留最近 4 条消息（2 轮对话）
                    .toList();
            messages.addAll(recentHistory);
        }

        messages.add(ChatMessage.user(question));

        return ChatRequest.builder()
                .messages(messages)
                .temperature(0.1D)
                .topP(0.3D)
                .thinking(false)
                .build();
    }


    private RewriteResult parseRewriteAndSplit(String raw) {
        try {
            // 移除可能存在的 Markdown 代码块标记
            String cleaned = LLMResponseCleaner.stripMarkdownCodeFence(raw);

            JsonElement root = JsonParser.parseString(cleaned);
            if (!root.isJsonObject()) {
                return null;
            }
            JsonObject obj = root.getAsJsonObject();
            String rewrite = obj.has("rewrite") ? obj.get("rewrite").getAsString().trim() : "";
            List<String> subs = new ArrayList<>();
            if (obj.has("sub_questions") && obj.get("sub_questions").isJsonArray()) {
                JsonArray arr = obj.getAsJsonArray("sub_questions");
                for (JsonElement el : arr) {
                    if (el.isJsonPrimitive() && el.getAsJsonPrimitive().isString()) {
                        String s = el.getAsString().trim();
                        if (StrUtil.isNotBlank(s)) {
                            subs.add(s);
                        }
                    }
                }
            }
            if (StrUtil.isBlank(rewrite)) {
                return null;
            }
            if (CollUtil.isEmpty(subs)) {
                subs = List.of(rewrite);
            }
            return new RewriteResult(rewrite, subs);
        } catch (Exception e) {
            log.warn("解析改写+拆分结果失败，raw={}", raw, e);
            return null;
        }
    }

    private List<String> ruleBasedSplit(String question) {
        // 兜底：按常见分隔符拆分
        List<String> parts = Arrays.stream(question.split("[?？。；;\\n]+"))
                .map(String::trim)
                .filter(StrUtil::isNotBlank)
                .collect(Collectors.toList());

        if (CollUtil.isEmpty(parts)) {
            return List.of(question);
        }
        return parts.stream()
                .map(s -> s.endsWith("？") || s.endsWith("?") ? s : s + "？")
                .toList();
    }
}
