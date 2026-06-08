import cv2
import numpy as np


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    return img.astype(np.float32)


def compute_fft(img):
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)

    magnitude = np.abs(fft_shifted)

    # normalize energy scale
    magnitude = np.log1p(magnitude)

    return magnitude


def radial_decay_score(magnitude):
    """
    Measures instability in radial energy decay.
    """
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    y, x = np.indices((h, w))

    r = np.sqrt((x - cx) ** 2 +
                (y - cy) ** 2)

    r = r.astype(np.int32)

    radial_mean = np.bincount(
        r.ravel(),
        weights=magnitude.ravel()
    )

    counts = np.bincount(r.ravel())

    radial_mean /= np.maximum(counts, 1)

    radial_mean = radial_mean[1:]

    # smooth decay expected
    gradient = np.diff(radial_mean)

    instability = np.std(gradient)

    # squash to 0–1
    score = np.tanh(instability * 10)

    return score


def periodic_peak_score(magnitude):
    """
    Detect unusual spectral spikes.
    """
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]

    r = min(h, w) // 8

    mask = (
        (x - cx) ** 2 +
        (y - cy) ** 2 > r**2
    )

    hf = magnitude[mask]

    mu = np.mean(hf)
    sigma = np.std(hf)

    z_scores = (
        (hf - mu) /
        (sigma + 1e-8)
    )

    strong_peaks = np.sum(z_scores > 3)

    score = strong_peaks / len(hf)

    return np.clip(score * 40, 0, 1)


def directional_anisotropy_score(magnitude):
    """
    Measures directional regularity.
    """
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    y, x = np.indices((h, w))

    theta = np.arctan2(
        y - cy,
        x - cx
    )

    bins = 36

    angular_energy = []

    for i in range(bins):
        low = -np.pi + (
            i * 2 * np.pi / bins
        )

        high = low + (
            2 * np.pi / bins
        )

        mask = (
            (theta >= low) &
            (theta < high)
        )

        angular_energy.append(
            np.mean(magnitude[mask])
        )

    angular_energy = np.array(
        angular_energy
    )

    anisotropy = np.std(
        angular_energy
    )

    return np.tanh(
        anisotropy * 3
    )


def spectral_entropy_score(magnitude):
    """
    Lower entropy is suspicious.
    """
    mag = magnitude.flatten()

    prob = mag / np.sum(mag)

    entropy = -np.sum(
        prob *
        np.log2(prob + 1e-12)
    )

    # normalize
    # natural images ~17–19
    score = np.clip(
        (18 - entropy) / 4,
        0,
        1
    )

    return score


def verify_image_authenticity(
    image_path,
    threshold=0.45
):
    try:
        img = load_image(
            image_path
        )

        magnitude = compute_fft(
            img
        )

        radial = radial_decay_score(
            magnitude
        )

        periodic = (
            periodic_peak_score(
                magnitude
            )
        )

        anisotropy = (
            directional_anisotropy_score(
                magnitude
            )
        )

        entropy = (
            spectral_entropy_score(
                magnitude
            )
        )

        # weighted mathematical ensemble
        final_score = (
            0.35 * radial +
            0.30 * periodic +
            0.20 * anisotropy +
            0.15 * entropy
        )

        print(
            f"\n--- {image_path} ---"
        )

        print(
            f"Radial Score: "
            f"{radial:.4f}"
        )

        print(
            f"Periodic Score: "
            f"{periodic:.4f}"
        )

        print(
            f"Anisotropy Score: "
            f"{anisotropy:.4f}"
        )

        print(
            f"Entropy Score: "
            f"{entropy:.4f}"
        )

        print(
            f"Final Score: "
            f"{final_score:.4f}"
        )

        if final_score > threshold:
            print(
                "Classification: "
                "LIKELY DEEPFAKE"
            )
            return True
        else:
            print(
                "Classification: "
                "LIKELY AUTHENTIC"
            )
            return False

    except Exception as e:
        print(f"Error: {e}")


verify_image_authenticity(
    "mark.jpg"
)