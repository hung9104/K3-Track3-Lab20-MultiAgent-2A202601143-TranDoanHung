# Thiết kế hệ thống Multi-Agent Research

## Bài toán

Hệ thống nhận một câu hỏi nghiên cứu, thu thập nguồn, phân tích bằng chứng và viết câu trả
lời cuối có trích dẫn. Cùng một truy vấn phải chạy được qua single-agent baseline và
multi-agent workflow để so sánh chất lượng, độ trễ và chi phí.

## Vì sao dùng multi-agent?

Một agent duy nhất phù hợp làm baseline nhưng phải đồng thời tìm kiếm, đánh giá nguồn và
viết bài, khiến prompt dài và khó xác định bước gây lỗi. Multi-agent tách các trách nhiệm có
đầu ra kiểm chứng được. Kiến trúc này chỉ có giá trị nếu benchmark cho thấy lợi ích về chất
lượng hoặc khả năng truy vết lớn hơn chi phí điều phối phát sinh.

## Vai trò agent

| Agent | Trách nhiệm | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Chọn bước tiếp theo và áp dụng điều kiện dừng | Toàn bộ `ResearchState` | Route tiếp theo trong `AgentName` | Route không hợp lệ, lặp vô hạn |
| Researcher | Tìm và tóm tắt nguồn liên quan | `request`, giới hạn nguồn | `sources`, `research_notes` | Search lỗi, không có nguồn, nguồn trùng |
| Analyst | So sánh bằng chứng và đánh giá độ tin cậy | `sources`, `research_notes` | `analysis_notes` | Phân tích thiếu nguồn hoặc không có bằng chứng |
| Writer | Viết câu trả lời cuối có trích dẫn | `request`, sources và các notes | `final_answer` | Citation không tồn tại, câu trả lời rỗng |

`Critic` có trong skeleton nhưng không thuộc workflow tối thiểu. Chỉ thêm vào khi benchmark
chứng minh bước kiểm duyệt riêng mang lại lợi ích.

## Shared state

`ResearchState` là nguồn dữ liệu duy nhất được truyền giữa các node:

| Field | Mục đích |
|---|---|
| `request` | Truy vấn, đối tượng người đọc và số nguồn tối đa |
| `iteration` | Đếm số lần route để áp dụng giới hạn vòng lặp |
| `route_history` | Giải thích thứ tự agent đã chạy |
| `sources` | Các nguồn đã chuẩn hóa để phân tích và trích dẫn |
| `research_notes` | Kết quả bàn giao từ Researcher |
| `analysis_notes` | Kết quả bàn giao từ Analyst |
| `final_answer` | Đầu ra cuối của Writer |
| `agent_results` | Kết quả từng agent, latency và token usage |
| `token_usage` | Tổng token toàn workflow để benchmark chi phí |
| `trace` | Các sự kiện có cấu trúc để debug hoặc gửi tới tracing backend |
| `errors` | Lỗi có agent, attempt và cờ retryable để quyết định fallback |

Các schema tại biên component cấm field thừa, loại bỏ khoảng trắng và kiểm tra giới hạn dữ
liệu. URL nguồn chỉ chấp nhận HTTP/HTTPS.

## Routing policy

```text
Chưa có sources                   -> Researcher
Có sources, chưa có analysis     -> Analyst
Có analysis, chưa có answer      -> Writer
Có final_answer                  -> END
iteration >= max_iterations      -> STOP với lỗi có cấu trúc
worker gặp lỗi retryable         -> retry tối đa max_retries
worker vẫn lỗi hoặc không retry  -> fallback/STOP
```

Mỗi lần route phải gọi `record_route`; mỗi agent hoàn thành phải gọi `add_agent_result` để
token được cộng vào tổng usage.

## Guardrails

- Max iterations: `6`, cấu hình bằng `MAX_ITERATIONS`.
- Timeout: `60` giây cho một thao tác ngoài, cấu hình bằng `TIMEOUT_SECONDS`.
- Retry: tối đa `2` lần với backoff ban đầu `1.0` giây.
- Fallback: search có thể dùng nguồn mock; lỗi không phục hồi được phải được ghi vào state và dừng rõ ràng.
- Validation: Pydantic kiểm tra input/output, route enum, URL, token không âm và output không rỗng.
- Secrets: agent chỉ nhận dependency/config đã tạo sẵn, không tự đọc environment variables.

## Kế hoạch benchmark

Chạy cùng tập truy vấn trong `configs/lab_default.yaml` qua baseline và multi-agent, tối
thiểu ba lượt cho mỗi phương án.

| Metric | Cách đo | Kỳ vọng |
|---|---|---|
| Latency | Wall-clock time | Multi-agent thường chậm hơn |
| Cost | Tổng token và giá provider | Multi-agent thường tốn hơn |
| Quality | Rubric 0-10 | Multi-agent tốt hơn ở câu hỏi cần nhiều nguồn |
| Citation coverage | Claim chính có nguồn / tổng claim chính | Multi-agent có coverage cao hơn |
| Failure rate | Số lượt lỗi / tổng lượt | Không vượt baseline đáng kể |

Kết luận cuối phải dựa trên số liệu; không mặc định multi-agent là phương án tốt hơn.
