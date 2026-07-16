## 1. Bài báo 1: A Survey of Air-to-Ground Propagation Channel Modeling for Unmanned Aerial Vehicles
* **Tác giả:** Wahab Khawaja, Ismail Guvenc, David W. Matolak, Uwe-Carsten Fiebig, và Nicolas Schneckenburger
* **Nơi công bố:** Tạp chí *IEEE Communications Surveys & Tutorials*, Tập 21, Số 3, Năm 2019.

### Khái quát chung
Tài liệu khảo sát (survey) toàn diện về các mô hình lan truyền kênh vô tuyến Air-to-Ground - AG phục vụ cho thiết bị bay không người lái (UAV). Bài viết tổng hợp các chiến dịch đo đạc thực tế, các mô hình fading quy mô lớn và nhỏ, các hạn chế hiện tại và định hướng nghiên cứu tương lai trong các kịch bản 5G và xa hơn.

### Các nội dung trọng tâm
* **Mô hình hóa Fading quy mô lớn (Large-Scale Fading):**
  * *Suy hao đường truyền (Path Loss):* Khác biệt lớn so với kênh mặt đất truyền thống, suy hao AG phụ thuộc nhiều vào độ cao của UAV, góc ngẩng (elevation angle) đối với trạm mặt đất, và đặc thù môi trường (đô thị, nông thôn).
  * *Xác suất có tầm nhìn thẳng (LoS Probability):* Độ cao lớn tăng xác suất LoS, giúp giảm suy hao nhưng lại làm tăng khả năng lan truyền nhiễu xuyên kênh.
  * *Shadowing do thân máy bay (Airframe Shadowing):* Xảy ra khi thân hoặc cánh quạt của UAV chặn đường truyền thẳng giữa anten và trạm mặt đất khi UAV quay hoặc nghiêng.
* **Mô hình hóa Fading quy mô nhỏ (Small-Scale Fading):**
  * Do có thành phần LoS mạnh, phân bố fading Ricean thường được sử dụng thay vì Rayleigh.
  * Phân tích hiện tượng phân tán trễ (delay dispersion) và dịch phổ Doppler do UAV di chuyển nhanh trong không gian 3D.
* **Mô hình kênh AG của 3GPP:** Đánh giá các mô hình kênh tiêu chuẩn hóa gần đây của 3GPP dành cho UAV và so sánh chúng với kênh di động tế bào truyền thống hoặc kênh vệ tinh.

## 2. Bài báo 2: Night-time Detection of UAVs Using Thermal Infrared Camera
* **Tác giả:** Petar Andraši, Tomislav Radišić, Mario Muštra, và Jurica Ivošević
* **Nơi công bố:** Tạp chí *Transportation Research Procedia*, Tập 28, Năm 2017, Trang 183–190.

### Khái quát chung
Nghiên cứu tập trung vào giải pháp phát hiện UAV cỡ nhỏ (sUAV) vào ban đêm bằng cách sử dụng các camera nhiệt hồng ngoại sóng dài (Long-Wave Infrared - LWIR) giá thành thấp. Đây là phương án thay thế hiệu quả khi các camera quang học (RGB) thông thường bị tê liệt trong điều kiện thiếu sáng hoặc sương mù.

***Nhận xét: Tầm nhìn và hình ảnh khá kém, nếu muốn nhìn rõ cần chi phí cao và yêu cầu lớn về phần cứng. Tuy nhiên có thể tham khảo tạm thời nếu lí tưởng hóa mô hình jammer.***

## 3. Bài báo 3: Distributed Extended Kalman Filtering Based Techniques for 3-D UAV Jamming Localization
* **Tác giả:** Waleed Aldosari, Muhammad Moinuddin, Abdulah Jeza Aljohani, và Ubaid M. Al-Saggaf
* **Nơi công bố:** Tạp chí *Sensors*, Năm 2020, Tập 20, Số 22, Mã bài báo 6405.

### Khái quát chung
Bài báo đề xuất phương pháp định vị và bám vết (tracking) một UAV đóng vai trò thiết bị gây nhiễu (jammer) di động trong không gian 3D. Thay vì sử dụng các thuật toán định vị tập trung phức tạp, tác giả đề xuất hai thuật toán phân tán dựa trên Bộ lọc Kalman mở rộng (Extended Kalman Filter - EKF) với sự hỗ trợ của các nút biên mạng (boundary nodes).

### Các thuật toán đề xuất
1. **DEKF (Distributed Extended Kalman Filter):** Chỉ sử dụng thông tin công suất tín hiệu gây nhiễu nhận được (Jammer Received Signal Strength - JRSS) tại một nút biên mạng duy nhất nằm gần jammer nhất.
2. **DEKF-DR (Distance Ratio aided DEKF):** Tích hợp thêm tỷ số khoảng cách (Distance Ratio) ước lượng từ một nút rìa (edge node) thứ hai để nâng cao độ chính xác định vị mà vẫn giữ thuật toán hoạt động phân tán.

### Kết quả chính
* Thực hiện mô phỏng quỹ đạo bay 3D của jammer và so sánh với phương pháp lọc Kalman mở rộng tập trung truyền thống (Centralized EKF - EKF-Centr).
* Thuật toán phân tán đề xuất (đặc biệt là DEKF-DR) đạt được hiệu năng định vị tiệm cận với phương pháp tập trung nhưng giảm đáng kể độ phức tạp tính toán và băng thông truyền thông tin trong mạng. Điều này giúp các nút mạng cảm biến công suất yếu có thể xử lý thời gian thực.

***Nhận xét: Khá khả quan, có thể tích hợp lên UAV.***

## 4. Bài báo 4: Geolocation and Tracking by TDOA Measurements Based on Space–Air–Ground Integrated Network
* **Tác giả:** Jinzhou Li, Shouye Lv, Ying Jin, Chenglin Wang, Yang Liu, và Shuai Liao
* **Nơi công bố:** Tạp chí *Remote Sensing*, Năm 2023, Tập 15, Số 1, Mã bài báo 44.

### Khái quát chung
Nghiên cứu đề xuất một phương pháp định vị thụ động (passive geolocation) và bám vết mục tiêu phát xạ RF dựa trên các phép đo Sai lệch thời gian đến (Time Difference of Arrival - TDOA) trong mạng tích hợp Không gian - Không trung - Mặt đất (Space-Air-Ground Integrated Network - SAGIN).

### Phương pháp nghiên cứu
* **Hệ thống quan sát SAGIN:** Tận dụng sự kết hợp giữa các vệ tinh (lớp Không gian), UAV/khí cầu (lớp Không trung), và các trạm cảm biến mặt đất (lớp Mặt đất) làm các nút thu nhận tín hiệu để định vị mục tiêu.
* **Mô hình toán học:** Biểu diễn vị trí và vận tốc của mục tiêu trong hệ tọa độ Địa tâm Địa cố định (Earth-Centered Earth-Fixed - ECEF).
* **Phân tích CRLB (Cramér-Rao Lower Bound):** Thiết lập giới hạn dưới lý thuyết của sai số định vị để đánh giá hiệu năng của bộ ước lượng.
* **Bộ ước lượng Maximum Likelihood (ML):** Đề xuất bộ ước lượng ML để tính toán đồng thời tọa độ 3D và vector vận tốc của mục tiêu dựa trên dữ liệu TDOA thu được từ các lớp quan sát.

### Đánh giá sai số hệ thống
Bài báo so sánh chi tiết ảnh hưởng của 4 nguồn sai số lớn đến độ chính xác định vị:
1. Sai số đồng bộ đồng hồ giữa các trạm quan sát (clock synchronization error).
2. Sai số về vị trí của bản thân các trạm quan sát (observer position bias).
3. Sai số về độ cao của mục tiêu (target elevation bias).
4. Sai số do vector vận tốc của mục tiêu không nằm ngang (non-horizontal target velocity).

***Nhận xét: Yêu cầu khá cao về phần cứng, cần thiết bị giám sát mặt đất hỗ trợ, không phù hợp mô hình bài toán.***

## 5. Bài báo 5: Efficient Localization of Directional Emitters via Joint Beampattern Estimation
* **Tác giả:** Fraser Williams, Akila Pemasiri, Dhammika Jayalath, Terrence Martin, và Clinton Fookes
* **Nơi công bố:** Bản thảo lưu trữ arXiv, Năm 2024 (arXiv:2411.04364v4).

### Khái quát chung
Bài báo giải quyết bài toán định vị trực tiếp (Direct Position Determination - DPD) cho các nguồn phát tín hiệu vô tuyến định hướng (directional RF emitters) – một thách thức lớn trong tác chiến điện tử. Các phương pháp định vị truyền thống thường giả định nguồn phát đẳng hướng (omnidirectional) và bị lỗi nghiêm trọng khi nguồn phát có tính định hướng cao do công suất nhận được (RSS) bị điều biến liên tục theo góc quay của búp sóng.

### Đóng góp kỹ thuật
* **Mô hình hóa búp sóng tổng quát:** Đề xuất một mô hình toán học tổng quát biểu diễn búp sóng định hướng thông qua 2 tham số: Hướng búp sóng ($\phi$) và Độ rộng nửa công suất (Half-power beamwidth $\beta$).
* **Định vị trực tiếp DPD tích hợp:** Kết hợp thông tin cường độ tín hiệu (RSS), Góc đến (AOA) và Sai lệch thời gian (TDOA) vào một hàm mục tiêu hợp lý cực đại (Maximum Likelihood) duy nhất.
* **Thuật toán tối ưu hóa luân phiên (Alternating Maximization Algorithm):** Giải quyết bài toán tìm kiếm tối ưu 4 chiều (2D tọa độ + 2D tham số búp sóng) bằng cách chia nhỏ thành các pha tối ưu hóa 2 chiều lặp lại luân phiên. Thuật toán hội tụ nhanh chóng chỉ sau 3–4 chu kỳ lặp.
* **Chỉ số không chắc chắn (Contrast-Expanded Half-Power Uncertainty Metric):** Đề xuất một hệ số đo lường vùng không chắc chắn để đánh giá độ tin cậy của kết quả định vị.

### Kết quả nổi bật
* Giảm sai số định vị từ 49% đến 61% so với các phương pháp định vị truyền thống chỉ sử dụng AOA-TDOA trong điều kiện SNR cực thấp (-10 dB).
* Chứng minh qua CRLB rằng các máy thu đặt ở rìa của búp sóng phát (nơi có gradient RSS lớn nhất) sẽ thu được thông tin định vị tốt nhất.

## 6. Bài báo 6: Measurement Based Statistical Channel Characterization of Air–to–Ground Path Loss Model at 446MHz for Narrow–Band Signals in Low Altitude UAVs
* **Tác giả:** Burak Ede, Serhan Yarkan, Ali Rza Ekti, Tunçer Baykaş, Hakan Ali Çırpan, và Ali Görçin
* **Nơi công bố:** Viện Nghiên cứu An toàn Thông tin (BİLGEM), Hội đồng Nghiên cứu Khoa học và Công nghệ Thổ Nhĩ Kỳ (TÜBİTAK), Năm 2018.

### Khái quát chung
Đo đạc thực địa và xây dựng mô hình suy hao đường truyền (Path Loss) thống kê cho liên kết AG ở dải tần 446 MHz sử dụng tín hiệu băng hẹp. Nghiên cứu hướng tới việc triển khai các trạm phát sóng di động băng hẹp gắn trên UAV tầm thấp để cứu hộ khẩn cấp (truyền thoại và tin nhắn ngắn) tại các khu vực xảy ra thiên tai khi hạ tầng viễn thông mặt đất bị phá hủy.

### Phương pháp nghiên cứu và Kết quả
* **Chiến dịch đo đạc thực tế:** Thiết kế hệ thống thu phát vô tuyến thực tế hoạt động ở tần số 446 MHz. UAV bay ở độ cao thấp trong các môi trường khác nhau để đo đạc cường độ tín hiệu nhận được tại trạm mặt đất.
* **Mô hình hóa kênh truyền:**
  * Trích xuất các tham số suy hao đường truyền thực tế (Path Loss Exponent) và độ lệch chuẩn của hiện tượng che bóng (Shadowing).
  * Phân tích các trạng thái chuyển đổi giữa Tầm nhìn thẳng (LoS), Tầm nhìn bị che khuất một phần (OLOS) và Không có tầm nhìn thẳng (NLoS).