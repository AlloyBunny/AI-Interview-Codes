# RoPE (Rotary Position Embedding, 旋转位置编码)

这是目前LLM最常用的位置编码，它编码token间的相对位置信息，作用是将位置信息注入 $Q$/$K$ 的内积，使 attention score 显式依赖 token 的相对位置。RePE只做乘性旋转变换，比起之前做加法操作的位置编码，它不容易破坏语义。

## 做法

### Step 0：变量声明

计算Attention Score的时候要用到$Q=XW_Q$，其中$X\in\mathbb{R}^{T\times d}$；$W_Q\in\mathbb{R}^{d\times d}$；$Q\in\mathbb{R}^{T\times d}$（为方便讨论，这里忽略batch维度$B$）。

考虑sequence中的第$p$个位置的token，有$Q_p\in\mathbb{R}^{d}$，同理也有$K_p\in\mathbb{R}^{d}$。

### Step 1：拆成二维向量

展开$Q_p$得到$Q_p=[Q_{p,0},\:Q_{p,1},\:Q_{p,2}\ldots,Q_{p,d-2},\:Q_{p,d-1}]$），

然后把向量按维度下标两两分组：$Q_{p,2i}$和$Q_{p,2i+1}$分到一组

### Step 2：做旋转

每一对维度都有一个频率：$\theta_i=10000^{-2i/d}$，

位置为$p$的的token，对应旋转角为$p\cdot\theta_i$

对每个维度$i$，进行旋转操作：$\begin{pmatrix}Q_{p,2i}^{\prime}\\Q_{p,2i+1}^{\prime}\end{pmatrix}=\begin{pmatrix}\cos(p\theta_i)&-\sin(p\theta_i)\\\sin(p\theta_i)&\cos(p\theta_i)\end{pmatrix}\begin{pmatrix}Q_{p,2i}\\Q_{p,2i+1}\end{pmatrix}$。这里可以看作把平面指标坐标系上$(Q_{p,2i}^{\prime}, Q_{p,2i+1}^{\prime})$这个向量逆时针旋转了$p\cdot\theta_i$角度，所以称为旋转操作。

我们把这个操作定义为$Q_p' = \text{RoPE}(Q_p, p)$，对每个token都执行RoPE操作，得到$Q'$和$K'$，用于替代原本的$Q$和$K$，这就完成了RoPE位置编码。

**（注：代码实现其实只涉及到这一步，如果不深究理论细节，后面的step 3和推导都可以不学）**

### Step 3：算Attention

计算Attention很简单，就是拿$Q'$和$K'$替代$Q$和$K$，Attention Score是$\mathrm{score}=\frac{Q^{\prime}(K^{\prime})^\top}{\sqrt{d}}$。

所以Attention Output是：$\text{Attention}(Q^{\prime},K^{\prime},V)=\mathrm{softmax}\left(\frac{Q^{\prime}(K^{\prime})^\top}{\sqrt{d}}\right)V$。

我们尝试把这个式子展开，由于RoPE的计算涉及到token位置，我们考虑计算$Q_p$和$K_q$的Attention Score，定义$R_{p}=\begin{pmatrix}\cos(p\theta)&-\sin(p\theta)\\\sin(p\theta)&\cos(p\theta)\end{pmatrix}$，最终将得到$\mathrm{score}=\frac{Q_p^\top(R_{q-p}K_q)}{\sqrt{d}}$。该结果的完整推导见下一节

（注：每个维度$i$的$\theta_i$不同，所以$R_{q-p}$也不同，这里为方便最终式子表达略写了）

## 如何推导？

上一节给出了最终Attention Output的答案，但没讲如何计算，本节讲这里的推导。我所知的推导方法有4种：

1. 直接推导（比较复杂，中间会用到多次余弦公式转换）
2. 利用$R$矩阵的性质简化推导
3. 利用几何直觉发现$(p-q)\theta$这个“相对旋转角度”
4. 从复数角度思考进行推导

我们这里主要讲方法2。

记$R(\theta)=\begin{pmatrix}\cos(\theta)&-\sin(\theta)\\\sin(\theta)&\cos(\theta)\end{pmatrix}$，$Q_p^{(i)}=\begin{pmatrix}Q_{p,2i}\\Q_{p,2i+1}\end{pmatrix}$，$K_q^{(i)}=\begin{pmatrix}K_{q,2i}\\K_{q,2i+1}\end{pmatrix}$，

考虑计算$Q_p$和$K_q$在第$i$维度块上的Attention Score为：$\text{score}=\langle R(p\theta_i)Q_p^{(i)},R(q\theta_i)K_q^{(i)}\rangle $。

而其中$R$矩阵有两个性质：

- $R(\theta)^T=R(-\theta)$
- $R(a)R(b)=R(a+b)$

其中性质1代入计算易得，性质二可以靠余弦公式推导得到，也可以通过$R$矩阵的旋转作用做感性理解（旋转a角度，再旋转b角度，那就是旋转了a+b角度）。利用这两个性质容易计算得到：
$$
\mathrm{score}=\langle R(p\theta_i)Q_p^{(i)},R(q\theta_i)K_q^{(i)}\rangle\\=(Q_p^{(i)})^TR(p\theta_i)^TR(q\theta_i)K_q^{(i)}\\=(Q_p^{(i)})^TR(-p\theta_i)R(q\theta_i)K_q^{(i)}\\=(Q_p^{(i)})^TR((q-p)\theta_i)K_q^{(i)}
$$
整合每个维度块的结果，就能得到上一节的$\mathrm{score}=\frac{Q_p^\top(R_{q-p}K_q)}{\sqrt{d}}$。

# YaRN（Yet another RoPE extension）

**注：YaRN的完整代码很长，有大量工程优化，没有必要全部搞懂，只需要大概理解：（1）它解决什么问题；（2）方法大致思路。**

motivation：RoPE 在训练时只见过最大长度$L_{train}$，那么如果推理时$L_{test}>L_{train}$（称为**外推**），就会出问题。

问题1：RoPE中低频部分旋转到的位置，模型在训练的时候没见过（用模块1解决）。

问题2：长context的attention计算中，softmax出来的结果可能太平或者太尖锐（用模块2解决）。

## 模块一：NTK-by-parts（分段频率缩放）

- 对于高频，不缩放，保持原样
- 对于中频，使用线性插值，平滑过渡
- 对于低频，使用线性缩放

注：个人感觉这个模块1是很工程化的东西，没必要死磕源码，知道问题咋来的，解决思路大概咋样就行。

## 模块二：attention temperature scaling（注意力温度缩放）

在attention计算的时候，额外除以一个$T$，变成$\mathrm{softmax}\left(\frac{QK^\top}{T\sqrt{d}}\right)V$，

$T$越小，softmax计算出来的logits就越“尖锐”；$T$越大softmax计算出来的logits就越“平滑”。

注：这个温度并不新东西，我当初学Attention的时候就学到过。放在这里是因为YaRN被设定为解决外推问题的方法，因为有两个问题，所以要两个模块分别解决。