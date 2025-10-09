import numpy as np


N = 11
k = (N-1) // 2
mu = 2

# D matrix
D = np.array([[0.]*2*N] * 2*N)

for i in range(k+1):
    D[2*i][2*i+1] = 1.
    D[2*(i+k)][2*(i+k)+1] = 1.
    D[2*i+1][2*i+1] = -1/mu
    D[2*(i+k)+1][2*(i+k)+1] = -1/mu
    for j in range(k):
        if i == 0:
            D[2*i+1][2*j+2] = 2 #dH_x/(mu*N)
            D[2*i+1][2*(j+k)+2] = 3 #dH_y/(mu*N)
        
        else:
            if i == j+1:
                D[2*i+1][2*j+2] = 8 #(Bx - Ax)/(mu*N)
                D[2*(i+k)+1][2*(j+k)+2] = 9 #(By - Ay)/(mu*N)
                
            else:
                D[2*i+1][2*j+2] = 4 #Bx(mu*N)
                D[2*(i+k)+1][2*(j+k)+2] = 5 #By(mu*N)
            
            D[2*i+1][2*(j+k)+2] = 6 #Cx/(mu*N)
            D[2*(i+k)+1][2*j+2] = 7 #Cy/(mu*N)

# for i in range(k+1):
#     D[2*i][2*i+1] = 1.
#     D[2*(i+k)][2*(i+k)+1] = 1.
#     D[2*i+1][2*i+1] = -1/mu
#     D[2*(i+k)+1][2*(i+k)+1] = -1/mu
#     for j in range(k):
#         if i == 0:
#             D[2*i+1][2*j+2] = 'dH_x/(mu*N)' #2 #dH_x/(mu*N)
#             D[2*i+1][2*(j+k)+2] = 'dH_y/(mu*N)' #3 #dH_y/(mu*N)
        
#         else:
#             if i == j+1:
#                 D[2*i+1][2*j+2] = '(Bx - Ax)/(mu*N)' #8 #(Bx - Ax)/(mu*N)
#                 D[2*(i+k)+1][2*(j+k)+2] = '(By - Ay)/(mu*N)' #9 #(By - Ay)/(mu*N)
                
#             else:
#                 D[2*i+1][2*j+2] = 'Bx(mu*N)' #4 #Bx(mu*N)
#                 D[2*(i+k)+1][2*(j+k)+2] = 'By(mu*N)' #5 #By(mu*N)
            
#             D[2*i+1][2*(j+k)+2] = 'Cx/(mu*N)' #6 #Cx/(mu*N)
#             D[2*(i+k)+1][2*j+2] = 'Cy/(mu*N)' #7 #Cy/(mu*N)

print(D)
# print(np.all(D == D2))