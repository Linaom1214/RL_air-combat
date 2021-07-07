import numpy as np
import random
import math
import random
'''这一版不考虑速度的改变'''
import numpy
#创建环境类
class AirCombat():
    def __init__(self):
        #初始化状态空间
        self.position_r = [130000,100000,3000]
        self.position_b = [130000,110000,3000]
        #初始化动作空间
    #执行一步动作，转换一次状态，返回回报值和下一个状态
        x_r = self.position_r[0]
        x_b = self.position_b[0]
        y_r = self.position_r[1]
        y_b = self.position_b[1]
        z_r = self.position_r[2]
        z_b = self.position_b[2]
        self.gamma_r = 0
        self.gamma_b = 0

        self.pusin_r = 0
        self.pusin_b = 180

        self.v_r = 250
        self.v_b = 250
        self.active_r = [self.v_r,self.gamma_r,self.pusin_r]
        self.active_b = [self.v_b,self.gamma_b,self.pusin_b]


    '''输入当前位置状态信息和选择后的动作信息，得到下一步的位置状态信息'''
    '''考虑敌机为简单的匀速直线运动'''
    def generate_next_position(self,position_r,position_b,action_now,action_b):
        x_r = position_r[0]
        x_b = position_b[0]
        y_r = position_r[1]
        y_b = position_b[1]
        z_r = position_r[2]
        z_b = position_b[2]

        v_r = action_now[0]
        gamma_r = action_now[1]
        pusin_r = action_now[2]

        v_b= action_b[0]
        gamma_b= action_b[1]
        pusin_b= action_b[2]

        x_r_ = v_r*math.cos(gamma_r*(3.141592653/180))*math.sin(pusin_r*(3.141592653/180))
        y_r_ = v_r*math.cos(gamma_r*(3.141592653/180))*math.cos(pusin_r*(3.141592653/180))
        z_r_ = v_r*math.sin(gamma_r*(3.141592653/180))

        x_b_ = v_b*math.cos(gamma_b*(3.141592653/180))*math.sin(pusin_b*(3.141592653/180))

        y_b_ = v_b*math.cos(gamma_b*(3.141592653/180))*math.cos(pusin_b*(3.141592653/180))

        z_b_ = v_b*math.sin(gamma_b*(3.141592653/180))

        x_r_next = x_r + x_r_
        y_r_next = y_r + y_r_
        z_r_next = z_r + z_r_

        x_b_next = x_b + x_b_
        y_b_next = y_b + y_b_
        z_b_next = z_b + z_b_

        position_r_next = [x_r_next,y_r_next,z_r_next]
        position_b_next = [x_b_next,y_b_next,z_b_next]

        return position_r_next,position_b_next

    '''输入当前的位置信息和速度角度等状态信息，得到其所对应的态势信息'''
    def generate_state(self,position_r,position_b,action_r,action_b):
        x_r = position_r[0]
        x_b = position_b[0]
        y_r = position_r[1]
        y_b = position_b[1]
        z_r = position_r[2]
        z_b = position_b[2]
        v_r= action_r[0]
        gamma_r= action_r[1]
        pusin_r= action_r[2]
        v_b= action_b[0]
        gamma_b= action_b[1]
        pusin_b= action_b[2]
        d = math.sqrt((x_r-x_b)**2+(y_r-y_b)**2+(z_r-z_b)**2)
        q_r = math.acos(((x_b-x_r)*math.cos(gamma_r*(3.141592653/180))*math.sin(pusin_r*(3.141592653/180))+(y_b-y_r)*math.cos(gamma_r*(3.141592653/180))*math.cos(pusin_r*(3.141592653/180))+(z_b-z_r)*math.sin(gamma_r*(3.141592653/180)))/d)
        q_r_ = q_r*(180/3.141592653)
        q_b = math.acos(((x_r-x_b)*math.cos(gamma_b)*math.sin(pusin_b)+(y_r-y_b)*math.cos(gamma_b)*math.cos(pusin_b)+(z_r-z_b)*math.sin(gamma_b))/d)
        q_b_ = q_b*(180/3.141592653)
        beta = math.acos(math.cos(gamma_r*(3.141592653/180))*math.sin(pusin_r*(3.141592653/180))*math.cos(gamma_b*(3.141592653/180))*math.sin(pusin_b*(3.141592653/180))+math.cos(gamma_r*(3.141592653/180))*math.cos(pusin_r*(3.141592653/180))*math.cos(gamma_b*(3.141592653/180))*math.cos(pusin_b*(3.141592653/180))+math.sin(gamma_r*(3.141592653/180))*math.sin(gamma_b*(3.141592653/180)))
        beta_ = beta*(180/3.141592653)
        delta_h = z_r-z_b
        delta_v2 = v_r**2-v_b**2
        v2 = v_r**2
        h = z_r
        taishi = [q_r_,q_b_,d,beta_,delta_h,delta_v2,v2,h]
        return taishi

    '''注意角度问题,尚未做出更改'''
    '''输入当前的状态以及对动作的选择，得到当前的状态对应动作的下一步状态'''
    def action(self,v_r,gamma_r,pusin_r,flag,choose):

        #向右为正方向
        gamma_r_increase = gamma_r + 10
        gamma_r_decrease = gamma_r - 10
        gamma_r_constant = gamma_r
        pusin_r_increase = pusin_r + 10
        pusin_r_decrease = pusin_r - 10
        pusin_r_constant = pusin_r

        action_1 = [v_r,gamma_r_increase,pusin_r_increase]
        action_2 = [v_r,gamma_r_increase,pusin_r_constant]
        action_3 = [v_r,gamma_r_increase,pusin_r_decrease]
        action_4 = [v_r,gamma_r_constant,pusin_r_increase]
        action_5 = [v_r,gamma_r_constant,pusin_r_constant]
        action_6 = [v_r,gamma_r_constant,pusin_r_decrease]
        action_7 = [v_r,gamma_r_decrease,pusin_r_increase]
        action_8 = [v_r,gamma_r_decrease,pusin_r_constant]
        action_9 = [v_r,gamma_r_decrease,pusin_r_decrease]
        if flag == True:
            return [action_1,action_2,action_3,action_4,action_5,action_6,action_7,action_8,action_9]
        else:
            if choose == 0:
                return action_1
            elif choose == 1:
                return action_2
            elif choose == 2:
                return action_3
            elif choose == 3:
                return action_4
            elif choose == 4:
                return action_5
            elif choose == 5:
                return action_6
            elif choose == 6:
                return action_7
            elif choose == 7:
                return action_8
            elif choose == 8:
                return action_9

    def action_b(self,action_b):
        v_b_ = action_b[0]
        gamma_b = action_b[1]
        pusin_b = action_b[2]
        action_b = [v_b_,gamma_b,pusin_b]
        return action_b

    def normalize(self,array):
        array[0] = array[0] / 200
        array[1] = array[1] / 200
        array[2] = array[2] / 20000
        array[3] = array[3] / 200
        array[4] = array[4] / 10000
        array[5] = array[5]
        array[6] = array[6] / 40000
        array[7] = array[7] / 10000
        return array
    #def normal(self,state_array):
    def position_clip(self,position_r_list,position_b_list):
        x_r = position_r_list[0]
        x_b = position_b_list[0]
        if x_r>(x_b+100) or x_r<(x_b-100):
            x_r = x_b+random.randint(0,10)
        position_r_list[0] = x_r
        return position_r_list
    def epsilon_greedy(self,prediction,epsilon):
        num = random.random()
        if num>epsilon:
            temp = np.argmax(prediction)
        else:
            temp = random.randint(0,8)
        return temp