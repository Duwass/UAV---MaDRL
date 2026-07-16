## 1. Bài báo 1: Smart Jamming Attacks in 5G New Radio: A Review
* **Tác giả:** Youness Arjoune và Saleh Faruque
* **Đơn vị:** Khoa Khoa học Máy tính và Kỹ thuật Điện, Đại học North Dakota, Grand Forks, ND, Hoa Kỳ
* **Năm công bố:** 2020 (arXiv:2009.05531v1)

### Khái quát chung
Bài báo cung cấp một cái nhìn tổng quan toàn diện về kiến trúc của tiêu chuẩn 5G New Radio (NR) do 3GPP công bố năm 2017, đồng thời đánh giá chuyên sâu các lỗ hổng của 5G-NR trước các loại hình tấn công gây nhiễu khác nhau và tổng hợp các giải pháp phát hiện/giảm thiểu nhiễu trong tài liệu khoa học hiện đại.

### Các mô hình tấn công gây nhiễu (Jamming Models)
Bài báo phân loại 6 dạng thiết bị gây nhiễu phổ biến:
1. **Regular Jammer (Gây nhiễu chặn/liên tục):** Phát tín hiệu nhiễu liên tục trên băng thông rộng mà không tuân theo giao thức MAC. Tiêu tốn nhiều năng lượng, dễ bị phát hiện bởi các hệ thống trinh sát vô tuyến, nhưng không yêu cầu thông tin tiên nghiệm (a'priori) về hệ thống đích.
2. **Delusive Jammer (Gây nhiễu giả mạo):** Phát các chuỗi bit hợp lệ giả mạo vào kênh truyền một cách liên tục nhằm đánh lừa bộ thu đợi ở trạng thái lắng nghe. Rất khó phát hiện do tín hiệu nhiễu giống hệt tín hiệu thật.
3. **Random Jammer (Gây nhiễu ngẫu nhiên):** Luân phiên giữa trạng thái hoạt động (active) và nghỉ (idle) để tiết kiệm năng lượng. Trong pha hoạt động, nó có thể gây nhiễu liên tục hoặc giả mạo.
4. **Responsive Jammer (Gây nhiễu phản hồi/phản ứng):** Chỉ phát tín hiệu nhiễu khi phát hiện thấy kênh truyền đang có tín hiệu hoạt động của người dùng hợp lệ. Rất tiết kiệm năng lượng và khó bị phát hiện hơn.
5. **Go-next Jammer (Gây nhiễu bám kênh):** Chỉ gây nhiễu một kênh tần số tại một thời điểm. Nếu máy phát đổi kênh (nhảy tần), kẻ tấn công sẽ phát hiện và nhảy theo sang kênh mới.
6. **Control Channel Jammer (Gây nhiễu kênh điều khiển):** Nhắm trực tiếp vào các kênh điều khiển (PDCCH, PBCH) nhằm ngăn chặn việc thiết lập kết nối trước khi truyền dữ liệu.

### Lỗ hổng vật lý của 5G-NR trước gây nhiễu chọn lọc (Smart Jamming)
* **Kênh PBCH (Physical Broadcast Channel):** Gửi thông tin quảng bá (MIB). PBCH được phân bổ cố định trong lưới tài nguyên thời gian-tần số (SSB) nên dễ bị gây nhiễu chọn lọc với chu kỳ làm việc thấp (shallow duty cycle).
* **Kênh PDCCH (Physical Downlink Control Channel):** Chứa tài nguyên CORESET được phân bổ cục bộ trong miền tần số (khác với LTE trải rộng toàn băng thông). Jammer có thể giải mã thông tin CORESET để gây nhiễu chọn lọc trên các sóng mang con cụ thể.
* **Kênh PUCCH (Physical Uplink Control Channel):** Mặc dù PUCCH có tùy chọn nhảy tần nội khe (intra-slot hopping), các thông số nhảy tần được công khai theo chuẩn 5G. PUCCH sử dụng mã cực (Polar Codes) có khả năng bảo vệ kém trước gây nhiễu.
* **Kênh PRACH (Random Access Channel):** Jammer có thể làm tràn kênh bằng các preamble không hợp lệ để làm quá tải thủ tục truy cập ngẫu nhiên của gNB.
* **Massive MIMO:** Jammer có thể tấn công vào pha ước lượng kênh truyền thông qua phát nhiễu phi công (pilot contamination) hoặc tấn công hạng kênh (channel rank attack).

### Các kỹ thuật phát hiện và giảm thiểu nhiễu (Anti-Jamming)
* **Kỹ thuật phát hiện:**
  * *Dựa trên ngưỡng (Threshold-based):* Giám sát các chỉ số chất lượng như PDR (Tỷ lệ phân phối gói), BER (Tỷ lệ lỗi bit), SNR (Tỷ lệ tín hiệu trên nhiễu). Phương pháp này đơn giản nhưng có tỷ lệ báo động giả cao.
  * *Dựa trên thống kê (Statistical-based):* Sử dụng dữ liệu lịch sử để tính toán các đặc trưng thống kê của tín hiệu bị gây nhiễu.
  * *Dựa trên Trí tuệ nhân tạo (ML/DL):* Random Forest, SVM, Deep Learning (DRL) giúp tự động phân loại trạng thái kênh truyền. Thách thức lớn là thiếu bộ dữ liệu (dataset) thực tế công khai để huấn luyện.
* **Kỹ thuật giảm thiểu:**
  * Phổ trải chuỗi trực tiếp (DSSS) và Phổ trải nhảy tần (FHSS - gồm nhảy tần chậm và nhảy tần nhanh).
  * Lý thuyết trò chơi (Game Theory) để tìm điểm cân bằng Nash giữa chiến thuật tránh nhiễu của người dùng và chiến thuật tấn công của jammer.
  * Kênh định thời (Timing channels), Suppression of Jammers (MIMO), và tối ưu hóa điều phối tài nguyên bằng Học sâu (Deep Learning).

## 2. Bài báo 2: Accumulate and Jam: Towards Secure Communication via A Wireless-Powered Full-Duplex Jammer
* **Tác giả:** Ying Bi và He Chen
* **Đơn vị:** Trường Kỹ thuật Điện và Thông tin, Đại học Sydney, Úc
* **Năm công bố:** 2016 (arXiv:1608.01848v1)

### Khái quát chung
Nghiên cứu đề xuất một giao thức bảo mật lớp vật lý (Physical Layer Security - PLS) mới có tên là **Tích lũy và Gây nhiễu (Accumulate-and-Jam - AnJ)** áp dụng cho một nút gây nhiễu thân thiện (friendly jammer) hoạt động ở chế độ Song công toàn phần (Full-Duplex - FD) và được cấp nguồn hoàn toàn bằng thu hoạch năng lượng vô tuyến (Wireless Energy Harvesting - WEH).

### Mô hình hệ thống và Thiết kế phần cứng
* **Các nút mạng:** Gồm 1 Nguồn (Source - S), 1 Đích (Destination - D), 1 Kẻ nghe lén thụ động (Eavesdropper - E) và 1 Nút gây nhiễu thân thiện (Jammer - J).
* **Cơ chế thu hoạch năng lượng:** Jammer $J$ không có nguồn pin sẵn mà sử dụng các khối chỉnh lưu để chuyển đổi tín hiệu RF thu được từ $S$ thành dòng điện một chiều (DC).
* **Hệ thống lưu trữ năng lượng hỗn hợp (Hybrid Energy Storage):** Do hoạt động song công toàn phần (vừa thu năng lượng vừa phát nhiễu đồng thời), $J$ được trang bị hai loại lưu trữ:
  * *PES (Primary Energy Storage):* Pin hóa học sạc lại được với mật độ năng lượng cao.
  * *SES (Secondary Energy Storage):* Siêu tụ điện (super-capacitor) với mật độ công suất cao phục vụ lưu trữ tạm thời trong pha phát sóng.

### Giao thức Accumulate-and-Jam (AnJ)
Dựa vào mức năng lượng tích lũy $\varepsilon[k]$ tại khối PES và điều kiện kênh truyền $C_{SD}$, hệ thống chuyển đổi linh hoạt giữa 2 chế độ hoạt động trong mỗi khe truyền dẫn:
1. **Dedicated Energy Harvesting (DEH - Thu hoạch năng lượng chuyên dụng):** Kích hoạt khi năng lượng của Jammer thấp hơn ngưỡng ($\varepsilon[k] < E_{th}$). Lúc này, $S$ chỉ phát tín hiệu mang năng lượng (không chứa thông tin) và $J$ sử dụng toàn bộ $N_J = N_r + N_t$ anten để thu hoạch tối đa năng lượng nạp vào PES. Nút đích $D$ giữ im lặng.
2. **Opportunistic Energy Harvesting (OEH - Thu hoạch năng lượng cơ hội):** Kích hoạt khi năng lượng $\varepsilon[k] \ge E_{th}$ và dung lượng kênh $C_{SD} \ge R_s$ (tốc độ bảo mật đích). Lúc này, $S$ truyền dữ liệu đến $D$. Jammer $J$ hoạt động ở chế độ song công: sử dụng $N_t$ anten để phát tín hiệu nhiễu nhân tạo (artificial noise) triệt tiêu tại $D$ (nằm trong không gian không - null space của kênh $J \to D$) nhằm phá hoại khả năng nghe lén của $E$, đồng thời sử dụng $N_r$ anten còn lại để thu hoạch năng lượng từ chính tín hiệu RF mà $S$ phát ra (overhearing).

***Nhận xét: Khó áp dụng cơ chế thu năng lượng lên jammer di động do yêu cầu cao về phần cứng.***

---

## 3. Bài báo 3: String Stability Analysis of Cooperative Adaptive Cruise Control under Jamming Attacks
* **Tác giả:** Amir Alipour-Fanid, Monireh Dabaghchian, Hengrun Zhang, và Kai Zeng
* **Đơn vị:** Đại học George Mason, Fairfax, VA, Hoa Kỳ
* **Năm công bố:** 2017

***Nhận xét: Phương pháp tối ưu năng lượng cho jammer bằng cách tìm vị trí gây nhiễu tốt nhất, tuy nhiên không áp dụng được vào cho mô hình muốn xây dựng***

---

## 4. Bài báo 4: Practical Trial for Low-Energy Effective Jamming on Private Networks With 5G-NR and NB-IoT Radio Interfaces
* **Tác giả:** Paweł Skokowski, Krzysztof Malon, Michał Kryk, Krzysztof Maślanka, Jan M. Kelner, Piotr Rajchowski, Jarosław Magiera
* **Đơn vị:** Học viện Kỹ thuật Quân sự Ba Lan (MUT) & Đại học Công nghệ Gdańsk, Ba Lan
* **Năm công bố:** 2024 (IEEE Access)

### Khái quát chung
Thử nghiệm thực tế (practical trials) về tấn công gây nhiễu thông minh, tiết kiệm năng lượng trên mạng riêng 5G SA (Standalone Private Network) và NB-IoT (Narrowband IoT). Mục tiêu là chứng minh tính khả thi của việc gắn thiết bị gây nhiễu công suất thấp trên UAV cỡ nhỏ tiếp cận trạm gốc để vô hiệu hóa liên lạc mà không bị phát hiện.

### Kịch bản tác chiến điện tử bằng UAV Jammer
* **UAV-based Jamming:** Đưa jammer tiếp cận sát trạm gốc gNB (đối với gây nhiễu đường lên - UL) hoặc UE (đối với gây nhiễu đường xuống - DL). Nhờ khoảng cách cực ngắn, công suất phát nhiễu cần thiết để áp đảo tín hiệu có ích giảm đi hàng chục dB (theo định luật suy hao khoảng cách).
* **Đồng bộ hóa bằng GNSS/PPS:** Thiết bị gây nhiễu (SDR USRP) đồng bộ thời gian thông qua xung PPS của GPS để khớp chính xác cấu trúc khung/khe của mạng mục tiêu, giúp phát nhiễu đúng thời điểm mà không cần phải lắng nghe liên tục, giải quyết hiện tượng lệch pha (timing drift).

### Kết quả thử nghiệm trên Mạng riêng 5G SA (Băng tần N77-TDD & N7-FDD)
* **Hệ thống thử nghiệm:** Mạng lõi 5G SA của Athonet, phần RAN xử lý bằng phần mềm của Amarisoft, anten USRP MIMO 2x2. Thiết bị gây nhiễu gồm máy tính nhúng Raspberry Pi 4 kết hợp USRP B200 mini.
* **Chiến thuật gây nhiễu:** Thay vì gây nhiễu chặn 20 MHz, jammer phát 4 kênh băng hẹp (62.5 kHz mỗi kênh, tổng 250 kHz).
* **Hiệu năng thực tế:**
  * *Chế độ FDD:* Gây nhiễu DL làm sụt giảm băng thông DL từ **36.9 Mbps xuống 0 Mbps (giảm 100%)**, trong khi UL hầu như không bị ảnh hưởng do hoạt động ở tần số độc lập khác.
  * *Chế độ TDD:* Do DL và UL dùng chung tần số theo các khe thời gian, việc phát nhiễu băng hẹp nhảy tần khiến giao thức quản lý tài nguyên vô tuyến RRC bị lỗi nghiêm trọng, dẫn đến ngắt kết nối hoàn toàn giữa điện thoại (Xiaomi Mi11) và gNB.
  * *Hiệu quả năng lượng:* Gây nhiễu chọn lọc băng hẹp mang lại **lợi ích năng lượng (Energy Gain) từ 19 dB đến 25 dB** so với gây nhiễu chặn diện rộng, giúp thiết bị hoạt động lâu hơn trên UAV.

### Kết quả thử nghiệm trên NB-IoT (Băng tần LTE 8 - 935.1 MHz)
* **Chiến thuật gây nhiễu chọn lọc NRS (NRS Jamming):** Tấn công tập trung vào các tài nguyên của tín hiệu dẫn đường NRS (Narrowband Reference Signal) nằm trong phân khung ♯0 (NPBCH) và phân khung ♯8/♯9.
* **Hiệu năng thực tế:**
  * Gây nhiễu chặn bao phủ toàn bộ 180 kHz hầu như không ảnh hưởng đến việc giải mã thông tin hệ thống (MIB) của NB-IoT do mã hóa kênh sửa sai rất tốt.
  * Ngược lại, **gây nhiễu chọn lọc NRS làm sai lệch quá trình ước lượng kênh truyền (CSI)** tại bộ thu của thiết bị đầu cuối. Kết quả là tỷ lệ giải mã thành công MIB trên kênh NPBCH sụt giảm mạnh **xuống dưới 50%** chỉ với tỷ lệ công suất đỉnh nhiễu trên tín hiệu là 3 dB.

***Nhận xét: Dù không có cơ chế thu năng lượng nhưng phương pháp này tiết kiệm năng lượng cho jammer, giúp kéo dài thời gian hoạt động nhiều lần. Hạn chế: khó áp dụng được trong mô hình quân sự bảo mật cao hoặc các mô hình không sử dụng tín hiệu NRS/DMRS***
