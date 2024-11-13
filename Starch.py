import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


df=pd.read_excel("f:\Project_Data\Reflectance Data (1).xlsx")

data_frame=pd.DataFrame({
'Mean_Adom_Koo_1':df.iloc[:,1:67].mean(axis=1),
'Mean_Adom_Koo_2':df.iloc[:,68:134].mean(axis=1),
'Mean_Adom_Koo_3':df.iloc[:,135:201].mean(axis=1),
'Mean_Adusa_1':df.iloc[:,202:268].mean(axis=1),
'Mean_Adusa_2':df.iloc[:,269:335].mean(axis=1),
'Mean_Adusa_3':df.iloc[:,336:402].mean(axis=1),
'Mean_Adutwumw_1':df.iloc[:,403:469].mean(axis=1),
'Mean_Adutwumw_2':df.iloc[:,470:536].mean(axis=1),
'Mean_Adutwunw_3':df.iloc[:,537:603].mean(axis=1),
'Mean_Delayman_1':df.iloc[:,604:669].mean(axis=1),
'Mean_Delayman_2':df.iloc[:,670:736].mean(axis=1),
'Mean_Delayman_3':df.iloc[:,737:803].mean(axis=1),
'Mean_GiversKoo_Caps_1':df.iloc[:,804:870].mean(axis=1),
'Mean_GiversKoo_Caps_2':df.iloc[:,871:937].mean(axis=1),
'Mean_GiversKoo_Caps_3':df.iloc[:,938:1004].mean(axis=1),
'Mean_Kingdom_1':df.iloc[:,1005:1071].mean(axis=1),
'Mean_Kingdom_2':df.iloc[:,1072:1138].mean(axis=1),
'Mean_Kingdom_3':df.iloc[:,1139:1205].mean(axis=1),
'Mean_Natural_Man_1':df.iloc[:,1206:1272].mean(axis=1),
'Mean_Natural_Man_2':df.iloc[:,1273:1339].mean(axis=1),
'Mean_Natural_Man_3':df.iloc[:,1340:1406].mean(axis=1),
'Mean_P-Away_1':df.iloc[:,1407:1473].mean(axis=1),
'Mean_P-Away_2':df.iloc[:,1474:1540].mean(axis=1),
'Mean_P-Away_3':df.iloc[:,1541:1607].mean(axis=1),
'Mean_Prosluv_1':df.iloc[:,1608:1674].mean(axis=1),
'Mean_Prosluv_2':df.iloc[:,1675:1741].mean(axis=1),
'Mean_Prosluv_3':df.iloc[:,1742:1808].mean(axis=1),
'Mean_Prostacure_1':df.iloc[:,1809:1875].mean(axis=1),
'Mean_Prostacure_2':df.iloc[:,1876:1942].mean(axis=1),
'Mean_Prostacure_3':df.iloc[:,1943:2009].mean(axis=1),
'Mean_Prostafit_1':df.iloc[:,2010:2076].mean(axis=1),
'Mean_Prostafit_2':df.iloc[:,2077:2143].mean(axis=1),
'Mean_Prostafit_3':df.iloc[:,2144:2210].mean(axis=1),
'Mean_Prostajoy_1':df.iloc[:,2211:2277].mean(axis=1),
'Mean_Prostajoy_2':df.iloc[:,2278:2344].mean(axis=1),
'Mean_Prostajoy_3':df.iloc[:,2345:2411].mean(axis=1),
'Mean_Prostarite_1':df.iloc[:,2412:2478].mean(axis=1),
'Mean_Prostarite_2':df.iloc[:,2479:2545].mean(axis=1),
'Mean_Prostarite_3':df.iloc[:,2546:2612].mean(axis=1),
'Mean_Taabea_1':df.iloc[:,2613:2679].mean(axis=1),
'Mean_Taabea_2':df.iloc[:,2680:2746].mean(axis=1),
'Mean_Taabea_3':df.iloc[:,2747:2813].mean(axis=1),
'Mean_Tinattet_1':df.iloc[:,2814:2880].mean(axis=1),
'Mean_Tinatte_2':df.iloc[:,2881:2947].mean(axis=1),
'Mean_Tinatte_3':df.iloc[:,2948:3014].mean(axis=1),
'Mean_Truman_1':df.iloc[:,3015:3081].mean(axis=1),
'Mean_Truman_2':df.iloc[:,3082:3148].mean(axis=1),
'Mean_Truman_3':df.iloc[:,3149:3215].mean(axis=1),
'Mean_URO500_1':df.iloc[:,3216:3282].mean(axis=1),
'Mean_URO_500_2':df.iloc[:,3283:3349].mean(axis=1),
'Mean_URO_500_3':df.iloc[:,3350:3416].mean(axis=1),
'Mean_W.G._1':df.iloc[:,3417:3483].mean(axis=1),
'Mean_W.G._2':df.iloc[:,3484:3550].mean(axis=1),
'Mean_W.G._3':df.iloc[:,3551:3617].mean(axis=1),
'Mean_ZaharaMan_1':df.iloc[:,3618:3684].mean(axis=1),
'Mean_ZaharaMan_2':df.iloc[:,3685:3751].mean(axis=1),
'Mean_ZaharaMan_3':df.iloc[:,3752:3818].mean(axis=1)
 })

#Mean_Adom_Koo1=np.array(df['Mean_Adom_Koo'])
Wavelength=np.array(df["Wavelength"])

plt.figure(figsize=(10, 6))
for column in df.columns:
    plt.plot(data_frame,Wavelength,label=column)
    
plt.xlabel('Frequency')
plt.ylabel("wavelength")
plt.title("Reflectance Data")
#plt.legend()
plt.show()
