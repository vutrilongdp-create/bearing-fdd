# Báo cáo phân tích kiểm chứng: Pipeline phát hiện và chẩn đoán lỗi vòng bi

**Dự án:** SIMPAC-2025-290 - Bearing Fault Detection and Diagnosis  
**Định hướng:** Tái lập có kiểm chứng và phân tích sai khác giữa bài báo, mã nguồn và kết quả thực nghiệm

---

## 1. Mục tiêu phân tích

Tài liệu này tổng hợp các nhận xét kỹ thuật thu được trong quá trình tái lập pipeline phát hiện và chẩn đoán lỗi vòng bi dựa trên hai công trình của Magadán và cộng sự. Trọng tâm của phân tích không phải là khẳng định bài báo gốc đúng hoặc sai tuyệt đối, mà là làm rõ:

- Các thành phần chính trong pipeline: MS2AE, Health Index (HI), ngưỡng phát hiện lỗi, FFP, tín hiệu sai khác, Kurtogram/Spectral Kurtosis, phổ bao và XAI.
- Các điểm khác biệt giữa mô tả trong bài báo, mã nguồn triển khai và kết quả khi chạy thực nghiệm.
- Ảnh hưởng của từng quyết định triển khai đến khả năng phát hiện lỗi, chẩn đoán thành phần lỗi và tính ổn định của hệ thống.

Cách tiếp cận phù hợp cho báo cáo nghiên cứu là **tái lập có kiểm chứng**. Điều này giúp đề tài thể hiện năng lực hiểu phương pháp, kiểm tra mã nguồn và đánh giá khách quan các yếu tố ảnh hưởng đến kết quả.

---

## 2. Phân tích các vấn đề kỹ thuật chính

### 2.1. Xác định FFP và nguy cơ cảnh báo sai

Trong bài toán phát hiện lỗi sớm, First Faulty Point (FFP) là thời điểm hệ thống bắt đầu xác nhận có dấu hiệu suy giảm. Về nguyên tắc, FFP không nên được xác định chỉ từ một điểm HI vượt ngưỡng đơn lẻ, vì tín hiệu rung có thể chứa nhiễu, dao động tải hoặc các biến động ngắn hạn không phản ánh lỗi vật lý thực sự.

Theo mô tả trong bài báo, FFP được xác định khi xuất hiện **năm giá trị HI liên tiếp vượt ngưỡng**. Quy tắc này có ý nghĩa kỹ thuật rõ ràng: nó giảm khả năng cảnh báo sai do các điểm ngoại lai hoặc nhiễu đơn lẻ. Trong quá trình tái lập, nếu đường HI có hiện tượng vượt ngưỡng rất sớm ở giai đoạn đầu vòng đời, cần kiểm tra thêm bằng các bằng chứng phụ trợ như phổ bao, tần số lỗi đặc trưng hoặc xu hướng suy giảm kéo dài trước khi kết luận đó là lỗi thật.

Đối với tập IMS-1, việc bài báo lựa chọn các khoảng mẫu muộn để minh họa có thể được hiểu là nhằm phân tích các giai đoạn suy giảm đã rõ ràng hơn. Tuy nhiên, nếu chưa có kiểm chứng đầy đủ trên toàn bộ đường HI và phổ tần số, không nên kết luận chắc chắn rằng các dao động sớm là lỗi thật hoặc chắc chắn là báo động giả. Cách viết phù hợp hơn là:

> Các điểm vượt ngưỡng sớm cần được đánh giá thận trọng vì có thể phản ánh nhiễu hoặc dao động vận hành. Việc xác nhận lỗi nên dựa trên điều kiện vượt ngưỡng liên tiếp và bằng chứng phổ tần số liên quan đến tần số lỗi vòng bi.

---

### 2.2. Ngưỡng phát hiện lỗi: PDF, Q-Q plot và bách phân vị thứ 95

Một điểm cần làm rõ là cách xác định ngưỡng phát hiện lỗi từ HI của mẫu khỏe mạnh. Theo bài báo, trước hết các giá trị HI của mẫu khỏe được kiểm tra giả định phân phối bằng Q-Q plot. Sau đó, hàm mật độ xác suất (PDF) của HI healthy được tính và **bách phân vị thứ 95** được chọn làm ngưỡng phát hiện lỗi.

Do đó, khi tái lập, cần phân biệt rõ hai cách hiểu khác nhau:

1. Tính bách phân vị thứ 95 trực tiếp trên các giá trị HI healthy.
2. Tính bách phân vị trên các `bin_centers` của histogram/PDF.

Hai cách tính này có thể tạo ra các ngưỡng khác nhau đáng kể. Nếu lấy percentile trên `bin_centers`, ngưỡng có thể bị dịch lên hoặc dịch xuống tùy theo cách chia histogram. Vì vậy, không nên mặc định rằng phương pháp dùng `bin_centers` là chính xác hơn hoặc vượt trội hơn. Ngược lại, cần xem đây là một khác biệt triển khai cần được kiểm chứng.

Trong báo cáo, nên trình bày theo hướng:

> Bài báo lựa chọn ngưỡng dựa trên bách phân vị thứ 95 của phân bố HI healthy. Trong quá trình tái lập, cần kiểm tra cách tính percentile trong mã nguồn vì việc tính trực tiếp trên HI hoặc trên các bin của histogram có thể làm thay đổi ngưỡng và ảnh hưởng đến thời điểm FFP. Do đó, đề tài đánh giá ảnh hưởng của lựa chọn ngưỡng đến nguy cơ cảnh báo sai và bỏ sót lỗi.

Cách viết này an toàn hơn so với việc khẳng định một phương pháp ngưỡng là tối ưu tuyệt đối. Về mặt thực nghiệm, ngưỡng thấp có thể giúp phát hiện sớm hơn nhưng dễ tăng false alarm; ngưỡng cao có thể giảm false alarm nhưng có nguy cơ bỏ sót lỗi giai đoạn sớm. Đây là một bài toán đánh đổi cần được đánh giá trên từng dataset.

---

### 2.3. Triển khai Kurtogram: phụ thuộc MATLAB và phương án Python

Kurtogram/Spectral Kurtosis được dùng để xác định dải tần cộng hưởng có tính xung kích cao, phục vụ bước lọc thông dải và phân tích phổ bao. Trong mã nguồn gốc, nếu bước tính Kurtogram phụ thuộc vào MATLAB engine, hệ thống có thể gặp một số bất lợi:

- Khó tái lập trên các máy không có MATLAB hoặc cấu hình MATLAB không tương thích.
- Tăng chi phí gọi chéo giữa Python và MATLAB.
- Khó triển khai trong môi trường web hoặc môi trường nhẹ.
- Dễ phát sinh lỗi môi trường khi chạy nhiều lần hoặc chạy trên Windows.

Việc thay thế bằng một phương án Python thuần, ví dụ dựa trên `scipy.signal.spectrogram` và `scipy.stats.kurtosis`, giúp giảm phụ thuộc môi trường và cải thiện khả năng tái lập. Tuy nhiên, cần lưu ý rằng phương án này là một triển khai Spectral Kurtosis theo hướng thực dụng, chưa chắc tương đương hoàn toàn với fast kurtogram hoặc cấu trúc 1/3-binary tree trong bài báo gốc.

Vì vậy, cách trình bày phù hợp là:

> Việc chuyển bước tính Kurtogram sang Python giúp hệ thống dễ triển khai và ổn định hơn trong môi trường thử nghiệm. Tuy nhiên, cần tiếp tục so sánh dải tần được chọn bởi phiên bản Python với Kurtogram gốc để đánh giá mức độ tương đồng và ảnh hưởng đến kết quả chẩn đoán.

Nếu có số liệu benchmark về thời gian xử lý, có thể đưa vào báo cáo. Nếu chưa có bảng đo cụ thể, không nên khẳng định các con số như “40 giây xuống dưới 5 giây” hoặc “tăng tốc 800%” như một kết luận chính thức.

---

### 2.4. Quy tắc khớp đỉnh phổ và harmonic matching

Sau khi có phổ bao, hệ thống cần đối chiếu các đỉnh phổ với tần số lỗi đặc trưng của vòng bi như BPFO, BPFI, BSF và FTF cùng các harmonic tương ứng. Đây là bước quan trọng để phân loại lỗi vòng ngoài, vòng trong, bi lăn hoặc vòng cách.

Trong thực tế, phổ tần số có thể chứa nhiều đỉnh do nhiễu, cộng hưởng hoặc điều kiện vận hành. Do đó, nếu chỉ cần một đỉnh gần tần số lý thuyết để kết luận lỗi, hệ thống có thể tăng false positive. Việc bổ sung các tiêu chí như:

- Ngưỡng biên độ tối thiểu so với đỉnh trội.
- Yêu cầu xuất hiện từ hai harmonic trở lên.
- Giới hạn khoảng sai số tần số khi so khớp.

có thể giúp tăng độ tin cậy của bước chẩn đoán.

Tuy nhiên, các tiêu chí này cũng tạo ra sự đánh đổi. Nếu yêu cầu quá nghiêm ngặt, hệ thống có thể bỏ sót lỗi ở giai đoạn sớm, khi chỉ một harmonic nổi bật còn các harmonic khác yếu hoặc bị nhiễu che khuất. Vì vậy, trong báo cáo nên viết:

> Việc bổ sung ngưỡng biên độ và yêu cầu nhiều harmonic giúp giảm nguy cơ nhận nhầm đỉnh nhiễu là tần số lỗi. Tuy nhiên, các tham số này cần được lựa chọn thận trọng vì chúng ảnh hưởng trực tiếp đến sự đánh đổi giữa false positive và false negative.

Cách diễn giải này phù hợp với tinh thần khoa học hơn so với việc khẳng định quy tắc harmonic có thể loại bỏ hoàn toàn chẩn đoán sai.

---

### 2.5. XAI dựa trên tương quan giữa HI và đặc trưng kỹ thuật

Trong project, XAI nên được trình bày theo hướng **giải thích dựa trên đặc trưng kỹ thuật và phân tích tương quan**, không phải các phương pháp hậu nghiệm như SHAP hoặc LIME.

Các đặc trưng miền thời gian có thể sử dụng gồm:

- RMS.
- Skewness.
- Kurtosis.
- Crest Factor.
- Shape Factor.
- Impulse Factor.
- Margin Factor.

Các đặc trưng miền tần số có thể sử dụng gồm:

- Biên độ tại shaft frequency.
- BPFO.
- BPFI.
- BSF.
- FTF.
- Các đặc trưng này có thể được xét trên cả phổ lọc và phổ không lọc.

Ma trận tương quan Pearson giữa HI và các đặc trưng này giúp đánh giá mức độ liên hệ giữa đầu ra của mô hình học sâu và các đại lượng vật lý quen thuộc trong chẩn đoán dao động. Nếu HI có tương quan cao với RMS, Kurtosis hoặc biên độ tại tần số lỗi, điều này gợi ý rằng HI phản ánh một phần xu hướng suy giảm vật lý của hệ thống.

Tuy nhiên, cần diễn giải kết quả tương quan thận trọng:

- Tương quan cao không chứng minh quan hệ nhân quả.
- Tương quan có thể bị ảnh hưởng bởi xu hướng tăng/giảm chung theo thời gian.
- Không nên khẳng định HI “chắc chắn” học được bản chất vật lý nếu chưa có phân tích bổ sung.

Cách viết khuyến nghị:

> Ma trận tương quan được sử dụng như một công cụ hỗ trợ diễn giải, giúp kiểm tra mối liên hệ giữa HI và các đặc trưng vật lý của tín hiệu rung. Kết quả này làm tăng tính minh bạch của pipeline, nhưng không biến mô hình học sâu thành một mô hình hoàn toàn “hộp trắng”.

---

## 3. Những điểm cần kiểm chứng bổ sung

Để phần phân tích có sức thuyết phục hơn trong báo cáo nghiên cứu, nên bổ sung các bảng hoặc hình sau:

1. **Bảng so sánh ngưỡng HI**
   - Ngưỡng tính trực tiếp từ HI healthy.
   - Ngưỡng tính từ histogram/bin centers nếu có.
   - FFP tương ứng với từng cách tính.

2. **Bảng kết quả phát hiện lỗi**
   - Dataset.
   - Khoảng mẫu.
   - Ngưỡng HI.
   - FFP.
   - Nhận xét: phát hiện đúng, phát hiện muộn, phát hiện sớm, hoặc không phát hiện.

3. **Bảng kết quả chẩn đoán lỗi**
   - Lỗi kỳ vọng.
   - Lỗi phát hiện.
   - Số harmonic khớp.
   - Nhận xét: exact, mixed, unknown, no FFP.

4. **Benchmark thời gian xử lý Kurtogram**
   - MATLAB engine nếu có.
   - Python implementation.
   - Cùng một mẫu, cùng máy, lặp lại nhiều lần.

5. **Bảng tương quan XAI**
   - Correlation giữa HI và RMS, Kurtosis, Crest Factor.
   - Correlation giữa HI và các đặc trưng BPFO/BPFI/BSF/FTF.
   - Tách theo từng dataset nếu có thể.

---

## 4. Kết luận

Quá trình tái lập cho thấy pipeline phát hiện và chẩn đoán lỗi vòng bi không chỉ phụ thuộc vào mô hình MS2AE, mà còn phụ thuộc mạnh vào nhiều quyết định triển khai trong toàn bộ chuỗi xử lý tín hiệu. Các yếu tố như healthy baseline, cách xác định ngưỡng HI, quy tắc FFP, phương pháp chọn dải tần Kurtogram và tiêu chí harmonic matching đều có thể làm thay đổi kết quả cuối cùng.

Vì vậy, đóng góp phù hợp của đề tài nên được trình bày là:

- Tái lập và hệ thống hóa pipeline từ các công trình nền tảng.
- Kiểm chứng các bước quan trọng trong pipeline trên dữ liệu IMS và XJTU-SY.
- Phân tích ảnh hưởng của các lựa chọn triển khai đến kết quả phát hiện và chẩn đoán.
- Giảm phụ thuộc môi trường bằng cách thay thế một số bước xử lý bằng Python.
- Tăng khả năng diễn giải bằng phân tích tương quan giữa HI và các đặc trưng kỹ thuật.

Không nên kết luận rằng hệ thống đã đạt cấp độ thương mại hoặc sẵn sàng triển khai công nghiệp nếu chưa có kiểm thử trên dữ liệu thực tế đa dạng, benchmark thời gian đầy đủ và đánh giá định lượng về false positive/false negative.

Một kết luận phù hợp hơn là:

> Đề tài đã tái lập và kiểm chứng pipeline phát hiện, chẩn đoán lỗi vòng bi dựa trên MS2AE, HI, Kurtogram và phổ bao. Kết quả phân tích làm rõ các yếu tố ảnh hưởng đến độ ổn định và khả năng chẩn đoán của hệ thống, đồng thời cung cấp cơ sở để phát triển các cải tiến tiếp theo theo hướng RUL, real-time monitoring và XAI nâng cao.
