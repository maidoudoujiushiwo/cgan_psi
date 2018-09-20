# CGAN_PSI
采用gan（对抗神经网络）选择适合的建模样本的方法。
       
       程序说明
a=gan_psi('G0','D0',LR_G = 0.001, LR_D = 0.001,ART_COMPONENTS =100)#实例化，ART_COMPONENTS是变量维度

#迭代

sess= tf.Session()  

sess.run(tf.global_variables_initializer())  

bachsize=len(m2)

a1=m1[['y']].values

a2=m2[['y']].values

#m1,m2分别为大样本（待选择样本）和小样本（标准样本）

for i in range(1000):

    ax=np.random.randint(len(m1),size=bachsize)

    d1=data1[ax]
    
    D1,D2=a.train_step(d1,data2,a2,a1[ax])
    
    if i%50==0:  
    
       print D1,D2

p=a.eval_step(data1,a1)#结果打分

#效果不错，筛选出大样本的25%，教原样本psi下降了40%，筛选出的坏账占比与标准样本更加接近（原大样本11%，标准4%，筛选结果5%）


       
