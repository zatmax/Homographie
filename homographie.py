import numpy as np
import cv2 as cv

res_reduc = 4
zoom = 0.8

p11 = np.array([1056, 1859])
p11 = np.int_(p11/res_reduc)
p21 = np.array([211, 1847])
p21 = np.int_(p21/4)

p12 = np.array([1183, 2006])
p12 = np.int_(p12/res_reduc)
p22 = np.array([372, 2027])
p22 = np.int_(p22/4)

p13 = np.array([1677, 1991])
p13 = np.int_(p13/res_reduc)
p23 = np.array([991, 2012])
p23 = np.int_(p23/4)

p14 = np.array([1783, 2266])
p14 = np.int_(p14/res_reduc)
p24 = np.array([1131, 2359])
p24 = np.int_(p24/4)

A = np.array([[-p11[0], -p11[1], -1, 0, 0, 0, p11[0]*p21[0], p11[1]*p21[0], p21[0]],
              [0, 0, 0, -p11[0], -p11[1], -1, p11[0]*p21[1], p11[1]*p21[1], p21[1]],
              [-p12[0], -p12[1], -1, 0, 0, 0, p12[0]*p22[0], p12[1]*p22[0], p22[0]],
              [0, 0, 0, -p12[0], -p12[1], -1, p12[0]*p22[1], p12[1]*p22[1], p22[1]],
              [-p13[0], -p13[1], -1, 0, 0, 0, p13[0]*p23[0], p13[1]*p23[0], p23[0]],
              [0, 0, 0, -p13[0], -p13[1], -1, p13[0]*p23[1], p13[1]*p23[1], p23[1]],
              [-p14[0], -p14[1], -1, 0, 0, 0, p14[0]*p24[0], p14[1]*p24[0], p24[0]],
              [0, 0, 0, -p14[0], -p14[1], -1, p14[0]*p24[1], p14[1]*p24[1], p24[1]],])

Vh = np.linalg.svd(A)[2]

H = Vh[-1,:]
H = H / H[8]
H_inv = H.reshape((3,3))
H = np.linalg.inv(H_inv)

I1 = cv.imread("D:/Documents/INFO5/VA54/1.jpg")
I1 = cv.resize(I1, (int(I1.shape[1]/res_reduc), int(I1.shape[0]/res_reduc)))
I2 = cv.imread("D:/Documents/INFO5/VA54/2.jpg")
I2 = cv.resize(I2, (int(I2.shape[1]/res_reduc), int(I2.shape[0]/res_reduc)))

new_res_x = int(I1.shape[1])
new_res_y = int(I1.shape[0])

I_warp = np.zeros((new_res_y, new_res_x*2, 3), dtype=np.uint8)

for i in range(I1.shape[0]):
    for j in range(I1.shape[1]):
        I_warp[i][j] = I1[i][j]
        
# Version 1
'''for i in range(I2.shape[0]):
    for j in range(I2.shape[1]):
        p = np.array([[j],
                      [i],
                      [1]])
        p_warp = H @ p
        p_warp = p_warp / p_warp[2]
        
        if p_warp[0] >= 0 and p_warp[0] < new_res_x*2 and p_warp[1] >= 0 and p_warp[1] < new_res_y:
            I_warp[int(p_warp[1])][int(p_warp[0])] = I2[i][j]'''
          
# Version 2
for i in range(I_warp.shape[0]):
    for j in range(int(I_warp.shape[1] / 2), I_warp.shape[1]):
        p = np.array([[j],
                      [i],
                      [1]])
        p_warp = H_inv @ p
        p_warp = p_warp / p_warp[2]
        
        if p_warp[0] >= 0 and p_warp[0] < new_res_x and p_warp[1] >= 0 and p_warp[1] < new_res_y:
            I_warp[i][j] = I2[int(p_warp[1])][int(p_warp[0])]
            
p1 = np.reshape(np.append(p21, 1), (-1,1))
p2 = np.reshape(np.append(p22, 1), (-1,1))
p3 = np.reshape(np.append(p23, 1), (-1,1))
p4 = np.reshape(np.append(p24, 1), (-1,1))

p1 = H @ p1
p1 = np.int_(p1/p1[2])
p2 = H @ p2
p2 = np.int_(p2/p2[2])
p3 = H @ p3
p3 = np.int_(p3/p3[2])
p4 = H @ p4
p4 = np.int_(p4/p4[2])

cv.circle(I_warp,(p1[0][0],p1[1][0]), 5, (0,0,255), -1)
cv.circle(I_warp,(p2[0][0],p2[1][0]), 5, (0,0,255), -1)
cv.circle(I_warp,(p3[0][0],p3[1][0]), 5, (0,0,255), -1)
cv.circle(I_warp,(p4[0][0],p4[1][0]), 5, (0,0,255), -1)

zoomed_image = cv.resize(I_warp, None, fx=zoom, fy=zoom, interpolation=cv.INTER_LINEAR)

cv.imshow("image", zoomed_image)
cv.waitKey(0)
cv.destroyAllWindows()