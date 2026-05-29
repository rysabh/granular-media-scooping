import cv2
import numpy as np

WIDTH, HEIGHT = 720, 480
FPS = 20
OUT_PATH = "demo.mp4"

out = cv2.VideoWriter(
    OUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (WIDTH, HEIGHT),
)

pile_center = np.array([180, 340], dtype=float)
dump_center = np.array([610, 160], dtype=float)

spoon_pos = np.array([60, 340], dtype=float)
particles = []


def draw_frame(spoon_pos, particles, phase, carrying=False, show_pile=True):
    frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

    cv2.putText(frame, phase, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)

    if show_pile:
        cv2.putText(frame, "PILE", tuple((pile_center + [-35, 60]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 80, 40), 2)
        rng = np.random.default_rng(1)
        for _ in range(180):
            pt = (pile_center + rng.normal(0, 18, size=2)).astype(int)
            cv2.circle(frame, tuple(pt), 3, (150, 100, 50), -1)

    # dump zone
    cv2.rectangle(
        frame,
        tuple((dump_center + [-65, 45]).astype(int)),
        tuple((dump_center + [65, 80]).astype(int)),
        (120, 120, 120),
        2,
    )
    cv2.putText(frame, "DUMP ZONE", tuple((dump_center + [-75, 110]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)

    # particles
    for p in particles:
        cv2.circle(frame, tuple(p.astype(int)), 6, (0, 0, 255), -1)

    # tool
    tool_color = (0, 180, 0) if carrying else (0, 0, 0)
    cv2.circle(frame, tuple(spoon_pos.astype(int)), 18, tool_color, -1)
    cv2.putText(frame, "TOOL", tuple((spoon_pos + [-30, -25]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, tool_color, 2)

    return frame


# 1. APPROACH
for t in range(60):
    target = pile_center + np.array([-75, 0])
    spoon_pos += (target - spoon_pos) * 0.08
    frame = draw_frame(spoon_pos, particles, "APPROACH", carrying=False, show_pile=True)
    out.write(frame)


# 2. SCOOP
for t in range(60):
    if t < 20:
        target = pile_center + np.array([-30, 0])
    elif t < 42:
        target = pile_center + np.array([10, 0])
    else:
        target = pile_center + np.array([10, -25])

    spoon_pos += (target - spoon_pos) * 0.14

    if 18 <= t < 45 and len(particles) < 45:
        for _ in range(3):
            particles.append(spoon_pos + np.random.randn(2) * 7)

    frame = draw_frame(spoon_pos, particles, "SCOOP", carrying=len(particles) > 0, show_pile=True)
    out.write(frame)


relative_particles = [p - spoon_pos for p in particles]


# 3. LIFT
for t in range(60):
    spoon_pos += np.array([0, -3.6])
    particles = [spoon_pos + r for r in relative_particles]
    frame = draw_frame(spoon_pos, particles, "LIFT", carrying=True, show_pile=True)
    out.write(frame)


# 4. TRANSPORT (pure horizontal, far from pile)
for t in range(80):
    spoon_pos += np.array([5.0, 0.0])
    particles = [spoon_pos + r for r in relative_particles]

    frame = draw_frame(
        spoon_pos,
        particles,
        "TRANSPORT",
        carrying=True,
        show_pile=True,
    )
    out.write(frame)


# 5. DUMP
falling_particles = [spoon_pos + r for r in relative_particles]

for t in range(80):
    spoon_pos += np.array([0.8, -0.2])

    new_particles = []
    for p in falling_particles:
        p = p + np.array([np.random.randn() * 3.0, 8.0])
        if p[1] < HEIGHT - 10:
            new_particles.append(p)
    falling_particles = new_particles

    frame = draw_frame(
        spoon_pos,
        falling_particles,
        "DUMP / RELEASING",
        carrying=False,
        show_pile=False,
    )
    out.write(frame)

out.release()
print(f"Saved {OUT_PATH}")