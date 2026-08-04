import seaborn as sns 
import matplotlib.pyplot as plt
from data import capa_df_clean 
from data import capa_df

sns.lineplot(data=capa_df_clean, x='Cycle', y='SOH', hue='Cell')
plt.xlabel('Cycle Number')
plt.ylabel('State of Health (SOH)')
plt.title('Capacity Fade vs Cycle Number')
plt.show()

