import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, FancyArrowPatch, Arc
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

OUT_DIR = "hinh_minh_hoa_tieng_viet"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "black": "#111111",
    "gray": "#4F4F4F",
    "light_gray": "#F3F4F6",
    "mid_gray": "#D1D5DB",
    "blue": "#2563EB",
    "blue_light": "#DBEAFE",
    "green": "#16A34A",
    "green_light": "#DCFCE7",
    "orange": "#F97316",
    "orange_light": "#FFEDD5",
    "yellow_light": "#FEF9C3",
    "purple": "#9333EA",
    "purple_light": "#F3E8FF",
    "red": "#EF4444",
    "red_light": "#FEE2E2",
}


def export(fig, name):
    png = os.path.join(OUT_DIR, f"{name}.png")
    svg = os.path.join(OUT_DIR, f"{name}.svg")
    pdf = os.path.join(OUT_DIR, f"{name}.pdf")

    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Đã lưu: {png}")
    print(f"Đã lưu: {svg}")
    print(f"Đã lưu: {pdf}")


def setup_ax(figsize=(16, 9), xlim=(0, 14), ylim=(0, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def title(ax, text, y, fontsize=20):
    ax.text(
        7, y, text,
        ha="center", va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=COLORS["black"]
    )


def box(ax, x, y, w, h, text, fc=COLORS["light_gray"], ec=COLORS["black"],
        fontsize=10.5, weight="normal", radius=0.10, lw=1.6):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.04,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc
    )
    p.set_path_effects([
        pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.13),
        pe.Normal()
    ])
    ax.add_patch(p)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=COLORS["black"],
        linespacing=1.2
    )
    return p


def arrow(ax, xy1, xy2, color=COLORS["black"], lw=1.8, linestyle="-",
          rad=0.0, mutation=14, alpha=1.0):
    a = FancyArrowPatch(
        xy1, xy2,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}"
    )
    ax.add_patch(a)
    return a


def draw_uav(ax, x, y, scale=1.0, body_color=COLORS["blue"]):
    body = FancyBboxPatch(
        (x - 0.34 * scale, y - 0.11 * scale),
        0.68 * scale,
        0.22 * scale,
        boxstyle="round,pad=0.02,rounding_size=0.07",
        fc=body_color,
        ec=COLORS["black"],
        lw=1.2
    )
    ax.add_patch(body)

    ax.plot([x - 0.78 * scale, x + 0.78 * scale], [y, y],
            color=COLORS["black"], lw=1.2)
    ax.plot([x, x], [y - 0.52 * scale, y + 0.52 * scale],
            color=COLORS["black"], lw=1.2)

    for dx, dy in [(-0.86, 0), (0.86, 0), (0, 0.60), (0, -0.60)]:
        ax.add_patch(Circle(
            (x + dx * scale, y + dy * scale),
            0.14 * scale,
            fc="white",
            ec=COLORS["black"],
            lw=1.1
        ))
        ax.plot(
            [x + dx * scale - 0.10 * scale, x + dx * scale + 0.10 * scale],
            [y + dy * scale, y + dy * scale],
            color=COLORS["black"],
            lw=1.0
        )


def draw_uav_with_label(ax, x, y, label, active=False):
    color = COLORS["blue"] if active else COLORS["green"]
    draw_uav(ax, x, y, scale=0.72 if active else 0.66, body_color=color)

    # Nhãn đặt phía trên UAV, không nằm trên đường truyền
    fc = COLORS["blue_light"] if active else COLORS["green_light"]
    ec = COLORS["blue"] if active else COLORS["green"]

    box(
        ax, x - 0.78, y + 0.70, 1.56, 0.50,
        label,
        fc=fc,
        ec=ec,
        fontsize=9.5,
        weight="bold",
        radius=0.08,
        lw=1.3
    )


def draw_iot(ax, x, y, color, label_text, sub_text):
    dev = FancyBboxPatch(
        (x - 0.38, y - 0.24),
        0.76, 0.48,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        fc=color,
        ec=COLORS["black"],
        lw=1.4
    )
    ax.add_patch(dev)

    ax.add_patch(Rectangle(
        (x - 0.18, y - 0.06),
        0.36, 0.14,
        fc="white",
        ec=COLORS["black"],
        lw=0.9
    ))

    ax.plot([x + 0.36, x + 0.60], [y + 0.16, y + 0.42],
            color=COLORS["black"], lw=1.0)
    ax.add_patch(Circle((x + 0.62, y + 0.44), 0.035,
                        fc=COLORS["black"], ec=COLORS["black"]))

    ax.add_patch(Rectangle(
        (x - 0.28, y - 0.43),
        0.21, 0.08,
        fc=COLORS["green_light"],
        ec=COLORS["green"],
        lw=1.0
    ))
    ax.add_patch(Rectangle(
        (x - 0.05, y - 0.43),
        0.25, 0.08,
        fc=COLORS["blue_light"],
        ec=COLORS["blue"],
        lw=1.0
    ))

    ax.text(
        x, y - 0.72,
        label_text,
        ha="center", va="center",
        fontsize=10.8,
        fontweight="bold"
    )
    ax.text(
        x, y - 0.98,
        sub_text,
        ha="center", va="center",
        fontsize=9.6,
        color=COLORS["gray"]
    )


def draw_jammer(ax, x, y, scale=1.0):
    tower = Polygon(
        [
            (x, y + 0.55 * scale),
            (x - 0.32 * scale, y - 0.40 * scale),
            (x + 0.32 * scale, y - 0.40 * scale)
        ],
        closed=True,
        fc=COLORS["red_light"],
        ec=COLORS["black"],
        lw=1.4
    )
    ax.add_patch(tower)

    ax.plot([x, x], [y + 0.40 * scale, y - 0.35 * scale],
            color=COLORS["black"], lw=1.1)
    ax.plot([x - 0.18 * scale, x + 0.18 * scale],
            [y + 0.03 * scale, y + 0.03 * scale],
            color=COLORS["black"], lw=1.1)
    ax.plot([x - 0.24 * scale, x + 0.24 * scale],
            [y - 0.18 * scale, y - 0.18 * scale],
            color=COLORS["black"], lw=1.1)

    for r in [0.35, 0.55, 0.75]:
        ax.add_patch(Arc(
            (x, y + 0.55 * scale),
            r * scale,
            r * scale,
            theta1=25,
            theta2=155,
            color=COLORS["red"],
            lw=1.3
        ))

    ax.text(
        x, y - 0.82 * scale,
        "Bộ gây nhiễu\ncố định",
        ha="center", va="center",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["red"],
        linespacing=1.1
    )


def fig1_system_model_vn_fixed():
    fig, ax = setup_ax(figsize=(16, 9), xlim=(0, 14), ylim=(0, 8))

    title(
        ax,
        "Mô hình hệ thống UAV hỗ trợ mạng IoT lai dưới tác động gây nhiễu",
        7.55,
        fontsize=20
    )

    # Khu vực chính của hệ thống: x từ 0 đến 10.5
    main_right = 10.25

    # Panel chú thích riêng bên phải
    panel_x = 10.70
    panel_w = 2.95
    box(
        ax, panel_x, 5.55, panel_w, 1.62,
        "Chú thích",
        fc="white",
        ec=COLORS["mid_gray"],
        fontsize=11,
        weight="bold",
        radius=0.08,
        lw=1.3
    )

    legend_y = 6.75
    legend_items = [
        (COLORS["blue"], "-", "Truyền chủ động / điều khiển"),
        (COLORS["orange"], "-", "Sóng mang / tán xạ"),
        (COLORS["purple"], "-", "Thiết bị lai"),
        (COLORS["red"], "--", "Tác động gây nhiễu"),
        (COLORS["green"], "-", "Thay ca UAV"),
    ]

    for i, (color, ls, text) in enumerate(legend_items):
        y = legend_y - i * 0.27
        ax.plot(
            [panel_x + 0.28, panel_x + 0.72],
            [y, y],
            color=color,
            lw=2.2,
            linestyle=ls
        )
        ax.text(
            panel_x + 0.86, y,
            text,
            ha="left", va="center",
            fontsize=9.4,
            color=COLORS["black"]
        )

    # Đường phân tầng
    ax.plot([0.40, main_right], [4.35, 4.35], "--", color="#A3A3A3", lw=1.3)
    ax.text(0.45, 4.60, "Tầng UAV", fontsize=12, fontweight="bold", color=COLORS["gray"])
    ax.text(0.45, 1.10, "Tầng mặt đất", fontsize=12, fontweight="bold", color=COLORS["gray"])

    # UAV
    draw_uav_with_label(ax, 2.55, 5.80, "UAV-1\nđang hoạt động", active=True)
    draw_uav_with_label(ax, 5.55, 5.80, "UAV-2\ndự phòng", active=False)
    draw_uav_with_label(ax, 8.20, 5.80, "UAV-3\ndự phòng", active=False)

    # Thay ca UAV: để chữ phía trên đường, không đè
    arrow(ax, (3.30, 5.80), (4.70, 5.80), color=COLORS["green"], lw=2.2)
    ax.text(
        4.00, 6.16,
        "thay ca khi pin yếu",
        ha="center", va="center",
        fontsize=10,
        color=COLORS["green"],
        bbox=dict(fc="white", ec="none", alpha=0.95)
    )

    arrow(ax, (6.30, 5.80), (7.42, 5.80), color=COLORS["green"], lw=2.2)
    ax.text(
        6.86, 6.16,
        "thay ca",
        ha="center", va="center",
        fontsize=10,
        color=COLORS["green"],
        bbox=dict(fc="white", ec="none", alpha=0.95)
    )

    # Thiết bị IoT
    draw_iot(ax, 2.15, 2.18, COLORS["yellow_light"], "Loại 1", "chỉ truyền chủ động")
    draw_iot(ax, 5.30, 2.18, COLORS["orange_light"], "Loại 2", "chỉ tán xạ thụ động")
    draw_iot(ax, 8.45, 2.18, COLORS["purple_light"], "Loại 3", "linh hoạt hai cơ chế")

    # Jammer đặt bên phải nhưng vẫn trong hệ thống, không chồng chú thích
    draw_jammer(ax, 9.80, 4.45, scale=0.78)

    # Điểm neo cho đường truyền, tránh đi qua nhãn UAV
    uav_down = (2.55, 5.28)
    uav_up = (2.38, 5.28)

    type1_top = (2.12, 2.70)
    type2_top = (5.30, 2.70)
    type3_top = (8.45, 2.70)

    # Đường xuống từ UAV
    arrow(ax, uav_down, (1.95, 2.70), color=COLORS["blue"], lw=2.0, rad=0.04)
    arrow(ax, uav_down, (5.12, 2.70), color=COLORS["orange"], lw=2.0, rad=-0.07)
    arrow(ax, uav_down, (8.25, 2.70), color=COLORS["purple"], lw=2.0, rad=-0.08)

    # Đường lên từ thiết bị, để lệch một chút khỏi đường xuống
    arrow(ax, (2.32, 2.70), uav_up, color=COLORS["blue"], lw=1.8, rad=-0.12)
    arrow(ax, (5.48, 2.70), uav_up, color=COLORS["orange"], lw=1.8, rad=0.15)
    arrow(ax, (8.65, 2.70), uav_up, color=COLORS["purple"], lw=1.8, rad=0.14)

    # Jamming: giảm số đường và để nét mảnh hơn
    jam_origin = (9.45, 4.38)
    arrow(ax, jam_origin, (2.15, 2.72), color=COLORS["red"], lw=1.25, linestyle="--", rad=0.05, alpha=0.9)
    arrow(ax, jam_origin, (5.30, 2.72), color=COLORS["red"], lw=1.25, linestyle="--", rad=0.03, alpha=0.9)
    arrow(ax, jam_origin, (8.45, 2.72), color=COLORS["red"], lw=1.25, linestyle="--", rad=-0.02, alpha=0.9)
    arrow(ax, jam_origin, (2.80, 5.18), color=COLORS["red"], lw=1.10, linestyle="--", rad=0.03, alpha=0.75)

    # Hộp thông tin phía dưới, cách xa nhãn thiết bị
    box(
        ax, 0.95, 0.26, 2.45, 0.66,
        "Trạng thái IoT\npin cₙ, hàng đợi qₙ",
        fc=COLORS["light_gray"],
        fontsize=10.0
    )
    box(
        ax, 4.05, 0.26, 2.62, 0.66,
        "Quyết định UAV\nchế độ, khe thời gian, thay ca",
        fc=COLORS["light_gray"],
        fontsize=10.0
    )
    box(
        ax, 7.25, 0.26, 2.45, 0.66,
        "Mục tiêu\nthông lượng cao, mất mát thấp",
        fc=COLORS["light_gray"],
        fontsize=10.0
    )

    # Ghi chú nhỏ thay vì nhãn trên mũi tên
    box(
        ax, 10.70, 3.65, 2.95, 1.05,
        "Ý nghĩa mô hình\nUAV phát sóng, thu dữ liệu,\nđiều phối thiết bị và thay ca.",
        fc=COLORS["light_gray"],
        ec=COLORS["mid_gray"],
        fontsize=9.2,
        radius=0.08,
        lw=1.2
    )

    box(
        ax, 10.70, 2.25, 2.95, 1.05,
        "Jamming cố định\nlàm giảm chất lượng kênh\nvà tăng dữ liệu lỗi.",
        fc=COLORS["red_light"],
        ec=COLORS["red"],
        fontsize=9.2,
        radius=0.08,
        lw=1.2
    )

    export(fig, "hinh1_mo_hinh_he_thong_fixed")


if __name__ == "__main__":
    fig1_system_model_vn_fixed()