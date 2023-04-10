import cv2  # OpenCV 모듈 임포트
import numpy as np  # Numpy 모듈 임포트
import sys  # 시스템 모듈 임포트


def read_images(argv1, argv2):
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    return img1, img2


def find_homography_dlt(pts1, pts2):
    n = pts1.shape[0]

    # 변환된 좌표들
    pts1_h = np.hstack((pts1, np.ones((n, 1))))
    pts2_h = np.hstack((pts2, np.ones((n, 1))))

    # DLT를 위한 좌표 매트릭스 구성
    A = []
    for i in range(n):
        x, y = pts1_h[i][:2]
        u, v = pts2_h[i][:2]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A)

    # SVD를 이용하여 homography 행렬 계산
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)

    return H


def get_points(img):
    points = []  # 대응점 리스트 (4개이상)
    cv2.imshow("img", img)  # 이미지 GUI창에 띄우기

    def select_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 마우스 버튼 클릭했을때
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # GUI에 점 표시
            cv2.imshow("img", img)

    cv2.setMouseCallback("img", select_point)
    cv2.waitKey(0)  # 아무 키 입력을 기다림
    cv2.destroyAllWindows()
    return np.array(points)


def make_mask(img1, img2, type):
    window_size = 1600
    height_panorama = img1.shape[0]
    width_panorama = img1.shape[1] + img2.shape[1]

    offset = int(window_size / 2)
    barrier = img1.shape[1] - int(window_size / 2)

    mask = np.zeros((height_panorama, width_panorama))
    if type == "left":
        mask[:, barrier - offset : barrier + offset] = np.tile(
            np.linspace(1, 0, 2 * offset).T, (height_panorama, 1)
        )
        mask[:, : barrier - offset] = 1
    else:
        mask[:, barrier - offset : barrier + offset] = np.tile(
            np.linspace(0, 1, 2 * offset).T, (height_panorama, 1)
        )
        mask[:, barrier + offset :] = 1

    return cv2.merge([mask, mask, mask])


def my_warpPerspective(img, H, size):
    # getPerspectiveTransform 함수를 이용하여 H 행렬 구하기
    H_inv = np.linalg.inv(H)
    cols, rows = size
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    ones = np.ones(rows * cols)
    coords = np.vstack([x.ravel(), y.ravel(), ones])
    coords_transformed = H_inv @ coords
    x_transformed, y_transformed, _ = np.split(coords_transformed, 3, axis=0)
    x_transformed = x_transformed / y_transformed
    y_transformed = y_transformed / y_transformed
    x_transformed = x_transformed.reshape(rows, cols).astype(np.float32)
    y_transformed = y_transformed.reshape(rows, cols).astype(np.float32)

    # warpAffine 함수를 이용하여 이미지 변환하기
    result = cv2.remap(img, x_transformed, y_transformed, cv2.INTER_LINEAR)

    return result


def my_warpPerspective(img, H, out_shape):
    h, w = img.shape[:2]
    out_h, out_w = out_shape

    # 아웃풋 이미지 생성
    out_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # 호모그래피 행렬의 역행렬 계산
    H_inv = np.linalg.inv(H)

    # 아웃풋 이미지의 각 픽셀에 대해 호모그래피를 적용
    for y_out in range(out_h):
        for x_out in range(out_w):

            # 호모그래피를 적용하여 입력 이미지의 좌표를 계산.
            p_out = np.array([x_out, y_out, 1]).reshape((3, 1))
            p_in = np.matmul(H_inv, p_out)
            x_in, y_in = int(p_in[0, 0] / p_in[2, 0]), int(p_in[1, 0] / p_in[2, 0])

            # 바운더리 체크 후 아웃풋 이미지에 입력 이미지의 픽셀값을 복사
            if x_in >= 0 and x_in < w and y_in >= 0 and y_in < h:
                out_img[y_out, x_out] = img[y_in, x_in]

    return out_img


def blending(H, img1, img2):
    height_panorama = img1.shape[0]
    width_panorama = img1.shape[1] + img2.shape[1]

    # 1. 왼쪽 파노라마 이미지 만들기
    panorama_left = np.zeros((height_panorama, width_panorama, 3))  # 왼쪽 파노라마 이미지
    mask_left = make_mask(img1, img2, "left")  # 왼쪽 이미지를 위한 마스크 생성
    panorama_left[0 : img1.shape[0], 0 : img1.shape[1], :] = img1
    cv2.imwrite("panorama_left.jpg", panorama_left)
    panorama_left *= mask_left
    cv2.imwrite("panorama_left_masked.jpg", panorama_left)

    # 2. 오른쪽 파노라마 이미지 만들기
    mask_right = make_mask(img1, img2, "right")
    panorama_right = my_warpPerspective(img2, H, (height_panorama, width_panorama))
    cv2.imwrite("panorama_right.jpg", panorama_right)
    panorama_right = (
        my_warpPerspective(img2, H, (height_panorama, width_panorama)) * mask_right
    )
    cv2.imwrite("panorama_right_masked.jpg", panorama_right)

    # 3. 합치기
    result = panorama_left + panorama_right

    # 4. 검정부분을 제거하여 리턴
    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    filtered_result = result[min_row:max_row, min_col:max_col, :]
    return [result, filtered_result]


def main(argv1, argv2):
    img1, img2 = read_images(argv1, argv2)  # cv 를 활용하여 이미지 일기
    points1 = get_points(img1)  # 1번 이미지(왼쪽)으로 부터 대응점 선택
    points2 = get_points(img2)  # 2번 이미지(오른쪽)으로 부터 대응점 선택
    H = find_homography_dlt(points1, points2)  # DLT 방식을 이용한 호모그래피 구하기
    [result, filtered_result] = blending(H, img1, img2)  # 변환행렬을 통하여 파노라마 이미지 생성
    cv2.imwrite("panorama.jpg", result)  # 출력1
    cv2.imwrite("panorama_filtered.jpg", filtered_result)  # 출력2


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except IndexError:
        print("Usage: python stitch.py /images/img1.jpg /images/img2.jpg")
