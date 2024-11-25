from camera import Camera
import time

def main():
    cam = Camera(fps=60)
    time.sleep(2)
    target_duration = 1.0 / 50.0
    samples = 300
    for _ in range(samples):
        start = time.time()
        cam.get_frame()
        end = time.time()
        if end - start < target_duration:
            time.sleep(target_duration - (end - start))
    # print(f"Average age of frames: {cam.age_sum / float(samples)}")
    print(cam.age_sum.value / float(samples))
    # time.sleep(20)
    # print("stopping")
    
    # time.sleep(20)
    # cam.save_frame()

if __name__ == "__main__":
    main()
