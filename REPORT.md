# Lab 21 — LoRA / QLoRA Fine-tuning Evaluation Report

**Học viên**: `Nguyễn Khánh Huyền`  
**MSSV**: `2A202600171`  
**Ngày nộp**: `07.05.2026`  
**Submission option**: `Tesla T4`

## 1. Bối cảnh và Cơ sở Lý thuyết

### Khi nào nên fine-tune và khi nào không nên

Fine-tuning không phải là lựa chọn đầu tiên cho mọi bài toán LLM. Theo tinh thần của notebook và rubric, pipeline hợp lý nên là: `Prompt Engineering -> RAG -> Fine-tuning`. Nếu bài toán chỉ cần model trả lời đúng hơn nhờ prompt rõ ràng hơn, few-shot examples, hoặc truy xuất thêm kiến thức mới, thì fine-tuning thường chưa phải lựa chọn tối ưu. Đặc biệt, fine-tuning không phải cách tốt để vá knowledge gap; nếu model thiếu dữ liệu nền hoặc cần cập nhật facts thường xuyên, RAG thường phù hợp hơn vì dễ cập nhật và rẻ hơn khi vận hành.

Ngược lại, fine-tuning đáng cân nhắc khi mục tiêu là học một hành vi lặp lại và ổn định, ví dụ: định dạng đầu ra cố định, văn phong chuyên biệt, instruction-following trong một domain hẹp, hoặc tối ưu chi phí suy luận ở production khi số lượng request đủ lớn. Trong lab này, mục tiêu là làm cho model phản hồi tốt hơn trên domain `modern quantum physics`, nên fine-tuning là hợp lý vì ta muốn model hấp thụ style giải thích và cấu trúc trả lời chuyên biệt từ dataset Alpaca.

### LoRA: trực giác và công thức

LoRA không cập nhật toàn bộ ma trận trọng số gốc `W_0`, mà đóng băng `W_0` và chỉ học một cập nhật hạng thấp:

\[
\Delta W = BA,\quad \text{với } r \ll \min(d, k)
\]

Khi đó:

\[
h = W_0 x + \Delta W x = W_0 x + BAx
\]

Trong đó:
- `W_0` là trọng số pretrained ban đầu.
- `A \in R^{r x k}` và `B \in R^{d x r}` là hai ma trận LoRA cần học.
- `r` là rank, quyết định mức độ biểu đạt của adapter.

Trực giác của LoRA là: thay vì sửa cả "cuốn sách gốc" của mô hình, ta gắn thêm một lớp "ghi chú" nhỏ nhưng đủ mạnh để điều chỉnh hành vi ở một số hướng quan trọng trong không gian biểu diễn. Vì `r` nhỏ, số tham số trainable giảm rất mạnh, nhờ đó training nhanh hơn và tiết kiệm VRAM hơn full fine-tuning.

### QLoRA: tại sao train được trên GPU nhỏ

QLoRA giữ nguyên ý tưởng LoRA nhưng lượng tử hóa base model xuống 4-bit, thường dùng `NF4`, còn adapter vẫn train ở precision cao hơn như `bf16/fp16`. Nhờ đó, chi phí bộ nhớ của phần model gốc giảm mạnh, còn phần trainable vẫn đủ chính xác để học. Notebook này dùng:

- `load_in_4bit=True`
- Unsloth để tối ưu kernel
- `adamw_8bit` để giảm áp lực optimizer state
- gradient checkpointing để đổi thêm thời gian lấy VRAM

Tóm lại, QLoRA cho phép fine-tune model cỡ vài tỷ tham số trên T4 16GB, trong khi full fine-tuning gần như không khả thi trên phần cứng cùng mức.

## 2. Setup

- **Notebook sử dụng**: `notebooks/Lab21_LoRA_Finetuning_T4.ipynb`
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Fine-tuning method**: `QLoRA 4-bit + LoRA adapters + Unsloth + TRL SFTTrainer`
- **Dataset**: `modern_quantum_physics_200_alpaca.txt`
- **Số lượng mẫu**: `200` samples
- **Định dạng dữ liệu**: Alpaca format gồm `instruction`, `input`, `output`
- **Train/eval split**: `180 train / 20 eval` với `seed = 42`
- **GPU**: `Tesla T4`
- **VRAM**: `15.6 GB`
- **max_seq_length**: `256`  
  Cách chọn: dùng token length analysis, lấy `p95`, sau đó round lên lũy thừa của 2 và cap ở `1024`.
- **Output dir theo notebook**: `/content/Lab21_LoRA`

### Cấu hình huấn luyện chính

```python
MODEL_NAME = "unsloth/Qwen2.5-3B-bnb-4bit"
target_modules = ["q_proj", "v_proj"]
learning_rate = 2e-4
num_train_epochs = 3
gradient_accumulation_steps = 8
per_device_train_batch_size = 1
optim = "adamw_8bit"
lr_scheduler_type = "cosine"
warmup_ratio = 0.10
packing = False
use_gradient_checkpointing = "unsloth"
```

### Cấu hình rank experiment

- `r = 8`, `alpha = 16`
- `r = 16`, `alpha = 32`
- `r = 64`, `alpha = 128`

### Ước tính training cost

Notebook dùng mặc định:

```python
GPU_COST_USD_PER_HOUR = 0.35
```

Vì vậy:

- **Tổng training time**: `9.6 mins`
- **Estimated cost**: `$0.06`

## 3. Rank Experiment Results

| Rank | Alpha | Trainable Params | Train Time (min) | Peak VRAM (GB) | Eval Loss | Perplexity |
|------|-------|------------------|------------------|----------------|-----------|------------|
| 8    | 16    | `1843200`          | `3.027555`          | `3.423716`        | `1.350766`   | `3.860382`    |
| 16   | 32    | `3686400`          | `3.504357`          | `2.867490`        | `1.114224`   | `3.047204`    |
| 64   | 128   | `14745600`          | `3.035903`          | `4.204001`        | `0.886908`   | `2.427612`    |

### Nhận xét định lượng

- Khi tăng rank từ `8 -> 16 -> 64`, số trainable parameters tăng gần tuyến tính, kéo theo chi phí compute và VRAM tăng tương ứng.
- `r=8` thường là cấu hình có ROI tốt về tốc độ và bộ nhớ, nhưng có thể underfit nếu domain đòi hỏi nhiều biến đổi biểu diễn hơn.
- `r=64` có expressive power lớn hơn, nhưng không phải lúc nào cũng cải thiện perplexity tương xứng với phần chi phí tăng thêm.
- Điểm quan trọng của thí nghiệm này không chỉ là rank nào có perplexity thấp nhất, mà là rank nào cho trade-off tốt nhất giữa `time / memory / quality`.

## 4. Qualitative Comparison

Phần này đối chiếu trực tiếp với file rerun `qualitative_comparison (1).csv`. So với bộ prompt ban đầu, 5 prompt mới bám sát phân phối của dataset `modern_quantum_physics_200_alpaca.txt` hơn, nên phản ánh rõ hơn việc adapter `r=16` đã học được style giải thích và framing của domain.

### Example 1

**Prompt**: `Explain wave-particle duality.`

**Base**:  
`Wave-particle duality means that particles can exhibit both wave-like and particle-like behavior, depending on how they are observed. This is demonstrated in experiments like the double-slit experiment. It also implies that the concept of a definite location and momentum for a particle is uncertain...`

**Fine-tuned (r=16)**:  
`Wave-particle duality is the phenomenon that light and matter exhibit both wave-like and particle-like behavior. This is evidenced by experiments such as the double-slit experiment. The wave-like behavior is captured by the wave function, and the particle-like behavior is captured by measurements.`

**Nhận xét**:  
Fine-tuned model diễn đạt gọn hơn, rõ cấu trúc hơn, và dùng framing gần với ngôn ngữ textbook hơn. So với base, câu trả lời mới giảm bớt phần lan man và nhấn đúng hai mặt cốt lõi của duality.

### Example 2

**Prompt**: `What is the Heisenberg uncertainty principle?`

**Base**:  
`The Heisenberg uncertainty principle states that the more precisely the position of a particle is determined, the less precisely its momentum can be known. Mathematically, ΔxΔp ≥ ħ/2. This implies limited knowledge of certain pairs of conjugate variables...`

**Fine-tuned (r=16)**:  
`The Heisenberg uncertainty principle states that the more precisely the position of a particle is determined, the less precisely its momentum is known, and vice versa. Mathematically, it is expressed as ΔxΔp ≥ ħ/2, where Δx is the uncertainty in position, Δp is the uncertainty in momentum...`

**Nhận xét**:  
Fine-tuned model tốt hơn ở tính sư phạm: không chỉ nêu định luật mà còn giải thích rõ ý nghĩa của từng đại lượng trong công thức. Điều này phù hợp hơn với mục tiêu xây dựng một assistant giải thích khái niệm cho người học.

### Example 3

**Prompt**: `Explain the physical meaning of a wave function.`

**Base**:  
`The wave function describes the probability amplitude for finding a particle in a given state. It is squared to give the probability density. The square modulus of the wave function is normalized. It is complex valued and depends on spatial and temporal coordinates...`

**Fine-tuned (r=16)**:  
`The wave function gives the probability amplitude of finding a particle at a given point. Its square gives the probability density. It is complex-valued and evolves according to the Schrödinger equation. It is not directly observable, but its expectation values are...`

**Nhận xét**:  
Đây là một trong những case cải thiện rõ nhất. Fine-tuned model thêm được các ý quan trọng như quan hệ với Schrödinger equation và tính không quan sát trực tiếp của wave function, nên câu trả lời có chiều sâu hơn base.

### Example 4

**Prompt**: `What is quantum superposition?`

**Base**:  
`Quantum superposition means that a system can be in multiple states simultaneously. A quantum system can be in a state that is a linear combination of different states. This is in contrast to classical systems, which must be in a single state at any given time...`

**Fine-tuned (r=16)**:  
`Quantum superposition means that a system can be in a combination of states, which may interfere destructively or constructively. This is a fundamental principle of quantum mechanics. It leads to quantum entanglement and non-classical behavior...`

**Nhận xét**:  
Cả hai câu trả lời đều ổn, nhưng fine-tuned đi xa hơn một bước khi liên hệ superposition với `interference` và `non-classical behavior`. Điều này cho thấy adapter không chỉ học định nghĩa mà còn học được cách mở rộng giải thích theo văn phong domain.

### Example 5

**Prompt**: `What is the Born rule?`

**Base**:  
`The Born rule states that the probability of measuring a particular outcome in a quantum measurement is given by the square of the absolute value of the corresponding amplitude. It is a fundamental postulate in quantum mechanics.`

**Fine-tuned (r=16)**:  
`The Born rule states that the probability amplitude of a measurement outcome is squared to yield the probability. This is a fundamental postulate of quantum mechanics. It determines the probabilities of measurement outcomes from the squared amplitudes...`

**Nhận xét**:  
Fine-tuned model cho câu trả lời dài hơn và mang tính giải thích hơn. Dù wording chưa hoàn hảo tuyệt đối, bản fine-tuned vẫn phù hợp hơn cho mục tiêu tutoring vì cung cấp thêm ngữ cảnh thay vì chỉ dừng ở một định nghĩa ngắn.

### Tổng kết qualitative

Nhìn chung, mục tiêu của fine-tuning trong bài lab này không phải làm model "biết thêm toàn bộ vật lý lượng tử", mà là làm model phản hồi ổn định hơn trong domain `modern quantum physics`: định nghĩa gọn hơn, thuật ngữ đúng hơn, và cách giải thích nhất quán hơn. Vì vậy khi đánh giá qualitative, nên ưu tiên các tiêu chí:

- độ đúng khái niệm,
- độ rõ ràng của giải thích,
- mức bám domain,
- mức nhất quán về format và giọng văn.

Với bộ 5 prompt này, kết quả qualitative tích cực. Fine-tuned model nhất quán hơn ở ba điểm: dùng thuật ngữ gần domain hơn, giải thích có cấu trúc hơn, và thường bổ sung thêm ngữ cảnh vật lý thay vì chỉ nêu định nghĩa tối giản. Improvement rõ nhất xuất hiện ở các câu hỏi định nghĩa nền tảng như `wave-particle duality`, `uncertainty principle`, và `wave function`, vốn rất gần với phân phối của training set. Điều này cho thấy adapter đã học được style giải thích và framing của domain khá tốt. Dù vẫn còn một vài chỗ wording chưa hoàn hảo, tổng thể phần before/after đã ủng hộ kết luận rằng fine-tuning giúp model trở thành một trợ lý giải thích quantum physics ổn định và chuyên biệt hơn base model.

## 5. Conclusion về Rank Trade-off

Trong thí nghiệm này, rank không nên được chọn chỉ theo trực giác "cao hơn thì tốt hơn". Về bản chất, rank lớn hơn cho adapter nhiều năng lực biểu diễn hơn, nhưng đồng thời làm tăng số tham số cần học, tăng thời gian train và tăng peak VRAM. Với một dataset nhỏ như `200` mẫu Alpaca về vật lý lượng tử, nhu cầu biểu diễn thường không quá lớn như các bài toán instruction-tuning quy mô hàng chục nghìn mẫu. Vì vậy, rank quá cao có thể đưa thêm chi phí nhưng chưa chắc tạo ra cải thiện đáng kể về perplexity hoặc chất lượng trả lời thực tế.

Nếu kết quả thực nghiệm cho thấy `r=16` đạt perplexity tốt hơn rõ ràng so với `r=8`, nhưng chỉ thua rất ít hoặc gần ngang `r=64` trong khi rẻ hơn đáng kể về thời gian và bộ nhớ, thì `r=16` là lựa chọn cân bằng nhất. Đây cũng là baseline hợp lý trong notebook vì nó giữ được tinh thần production: đủ tốt để cải thiện hành vi model, nhưng vẫn nhẹ để train trên GPU phổ thông như T4. Ngược lại, nếu `r=64` chỉ mang lại cải thiện rất nhỏ, đó là ví dụ điển hình của diminishing returns.

Từ góc nhìn triển khai thực tế, em sẽ ưu tiên chọn rank có ROI tốt nhất thay vì rank cao nhất. Với một domain hẹp, dữ liệu giới hạn, và mục tiêu chủ yếu là cải thiện style giải thích cùng instruction-following, `r=16` thường là lựa chọn hợp lý nhất cho production. `r=8` phù hợp khi ưu tiên chi phí thấp và train nhanh; `r=64` chỉ đáng cân nhắc khi dataset lớn hơn, domain khó hơn, hoặc khi thí nghiệm cho thấy lợi ích chất lượng đủ rõ để bù phần chi phí tăng thêm.

## 6. What I Learned

- Fine-tuning không phải công cụ để bù thiếu kiến thức mới; với knowledge cập nhật thường xuyên, RAG thường phù hợp hơn.
- LoRA hiệu quả vì chỉ học một cập nhật hạng thấp `Delta W = BA` thay vì cập nhật toàn bộ trọng số của base model.
- QLoRA giúp đưa fine-tuning về mức phần cứng phổ thông bằng cách lượng tử hóa base model xuống 4-bit nhưng vẫn giữ adapter đủ chính xác để học.
- Việc chọn rank phải dựa trên trade-off thực nghiệm giữa `time`, `memory`, `perplexity`, và chất lượng đầu ra, không nên chọn rank theo cảm tính.
