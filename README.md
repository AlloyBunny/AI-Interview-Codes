一些用于互联网算法岗面试复习用的常见手撕代码

Cross-Attention:
Q = X_dec W_Q
K = H_enc W_K
V = H_enc W_V

PPO、GAE、GRPO
https://chatgpt.com/c/69b12ceb-5298-839e-85a2-076106fcd29a
https://chatgpt.com/share/69b24f2f-5c18-8003-9cd6-8db5650f8b11

为啥KL散度是D_{KL}(P||Q)=\sum_xP(x)\log\frac{P(x)}{Q(x)}？
意思是：用 Q 去近似 P，会损失多少信息

腾讯实习
https://chatgpt.com/c/69b2c08e-b064-8399-8e44-84d901df488a

以 OpenClaw 为例介绍 AI Agent 的运作原理
https://www.bilibili.com/video/BV1UqPQzXEmy/?vd_source=506760f143dfb492788260da660f3d42

【【闪客】一口气拆穿Skill/MCP/RAG/Agent/OpenClaw底层逻辑】 https://www.bilibili.com/video/BV1ojfDBSEPv/?share_source=copy_web&vd_source=7e010f16494955c14fbe8ca810a7c74e

【Agent Runtime】
while task_not_done:
    response = LLM(context)
    if response == tool_call:
        run_tool()
    update_memory()
    append_to_context()