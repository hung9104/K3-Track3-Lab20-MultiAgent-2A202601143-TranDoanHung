# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

Mỗi nhóm trả lời 2 câu:

1. Case nào nên dùng multi-agent? Vì sao?

   Nên dùng multi-agent cho nhiệm vụ phức tạp có thể tách thành các vai trò chuyên biệt,
   chẳng hạn nghiên cứu nhiều nguồn, đối chiếu bằng chứng, viết báo cáo có trích dẫn và
   kiểm duyệt chất lượng. Trong dự án này, Researcher thu thập nguồn, Analyst đánh giá
   bằng chứng, Writer tổng hợp câu trả lời và Critic kiểm tra citation trước khi kết thúc.
   Cách làm này phù hợp khi chất lượng, khả năng truy vết và việc xác định bước gây lỗi
   quan trọng hơn tốc độ phản hồi. Shared state và trace cũng giúp kiểm tra rõ agent nào
   tạo ra dữ liệu nào, thay vì để toàn bộ quá trình nằm trong một prompt dài.

2. Case nào không nên dùng multi-agent? Vì sao?

   Không nên dùng multi-agent cho câu hỏi đơn giản, tác vụ có một bước rõ ràng, yêu cầu
   phản hồi thời gian thực hoặc có ngân sách token thấp. Ví dụ: phân loại một câu, chuẩn
   hóa một trường dữ liệu, dịch một đoạn ngắn hoặc trả lời kiến thức trực tiếp thường chỉ
   cần single-agent. Việc thêm Supervisor và nhiều worker làm tăng số lần gọi LLM, latency,
   chi phí và số điểm có thể thất bại; handoff còn có thể làm mất ngữ cảnh. Chỉ nên chọn
   multi-agent khi benchmark trên cùng tập truy vấn cho thấy lợi ích về quality, citation
   coverage hoặc failure rate đủ lớn để bù cho phần chi phí và độ phức tạp tăng thêm.
