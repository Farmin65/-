"""
Модуль для визуализации результатов анализа
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import FIGURES_DIR


class Visualizer:
    """Класс для создания визуализаций"""
    
    def __init__(self, data):
        self.data = data
        self.set_style()
        
    def set_style(self):
        """Настройка стиля графиков"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
    def plot_anomaly_distribution(self, anomaly_mask):
        """Визуализация распределения аномалий"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Detour ratio распределение
        ax1 = axes[0, 0]
        ax1.hist(self.data['detour_ratio'][~anomaly_mask], bins=50, alpha=0.7, 
                label='Нормальные', color='green', density=True)
        ax1.hist(self.data['detour_ratio'][anomaly_mask], bins=20, alpha=0.7, 
                label='Аномальные', color='red', density=True)
        ax1.axvline(x=1.5, color='orange', linestyle='--', label='Порог (1.5)')
        ax1.set_xlabel('Коэффициент отклонения')
        ax1.set_ylabel('Плотность')
        ax1.set_title('Распределение коэффициента отклонения маршрута')
        ax1.legend()
        
        # 2. Скорость vs Detour ratio
        ax2 = axes[0, 1]
        scatter = ax2.scatter(self.data['detour_ratio'], self.data['avg_speed_kph'], 
                            c=anomaly_mask, cmap='RdYlGn_r', alpha=0.6, s=30)
        ax2.set_xlabel('Коэффициент отклонения')
        ax2.set_ylabel('Средняя скорость (км/ч)')
        ax2.set_title('Зависимость скорости от отклонения маршрута')
        plt.colorbar(scatter, ax=ax2, label='Аномалия')
        
        # 3. Стоимость по городам (только аномалии)
        ax3 = axes[1, 0]
        anomaly_data = self.data[anomaly_mask]
        if len(anomaly_data) > 0:
            city_anomalies = anomaly_data.groupby('origin_city').size().sort_values(ascending=True)
            city_anomalies.plot(kind='barh', ax=ax3, color='coral')
            ax3.set_xlabel('Количество аномалий')
            ax3.set_title('Аномалии по городам отправления')
        else:
            ax3.text(0.5, 0.5, 'Нет аномалий', ha='center', va='center')
            ax3.set_title('Аномалии по городам отправления')
        
        # 4. Временной ряд аномалий
        ax4 = axes[1, 1]
        daily_anomalies = self.data.groupby(self.data['date'].dt.date)['is_anomaly'].mean() * 100
        daily_anomalies.plot(ax=ax4, color='purple', marker='o', markersize=3)
        ax4.set_xlabel('Дата')
        ax4.set_ylabel('Доля аномалий (%)')
        ax4.set_title('Динамика аномалий во времени')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'anomaly_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    def plot_route_efficiency(self):
        """Визуализация эффективности маршрутов"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Эффективность по городам
        ax1 = axes[0]
        city_efficiency = self.data.groupby('origin_city')['route_efficiency'].mean().sort_values()
        colors = ['red' if x < 0.7 else 'orange' if x < 0.85 else 'green' 
                 for x in city_efficiency.values]
        city_efficiency.plot(kind='barh', ax=ax1, color=colors)
        ax1.axvline(x=0.85, color='orange', linestyle='--', alpha=0.7, label='Хорошо (0.85)')
        ax1.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Критично (0.7)')
        ax1.set_xlabel('Эффективность маршрута (1 / detour_ratio)')
        ax1.set_title('Эффективность маршрутов по городам')
        ax1.legend()
        
        # Boxplot по дням недели
        ax2 = axes[1]
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_efficiency = [self.data[self.data['day_of_week'] == day]['route_efficiency'].values 
                         for day in day_order]
        bp = ax2.boxplot(day_efficiency, labels=[d[:3] for d in day_order], patch_artist=True)
        for box in bp['boxes']:
            box.set_facecolor('lightblue')
        ax2.axhline(y=0.85, color='orange', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Эффективность маршрута')
        ax2.set_title('Эффективность по дням недели')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'route_efficiency.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    def plot_cost_analysis(self):
        """Анализ стоимости перевозок"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Стоимость vs расстояние
        ax1 = axes[0]
        normal = self.data[~self.data['is_anomaly']]
        anomalies = self.data[self.data['is_anomaly']]
        
        ax1.scatter(normal['optimal_distance_km'], normal['cost_rub'], 
                   alpha=0.5, s=20, label='Нормальные', color='green')
        ax1.scatter(anomalies['optimal_distance_km'], anomalies['cost_rub'], 
                   alpha=0.7, s=50, label='Аномальные', color='red', marker='x')
        ax1.set_xlabel('Оптимальное расстояние (км)')
        ax1.set_ylabel('Стоимость перевозки (руб)')
        ax1.set_title('Зависимость стоимости от расстояния')
        ax1.legend()
        
        # Средняя стоимость по городам
        ax2 = axes[1]
        city_cost = self.data.groupby('origin_city')['cost_rub'].mean().sort_values()
        city_cost.plot(kind='bar', ax=ax2, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Город отправления')
        ax2.set_ylabel('Средняя стоимость (руб)')
        ax2.set_title('Средняя стоимость перевозки по городам')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'cost_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    def create_dashboard(self, anomaly_mask):
        """Создание дашборда с ключевыми метриками"""
        fig = plt.figure(figsize=(16, 10))
        
        # Основные метрики
        total_shipments = len(self.data)
        anomaly_count = anomaly_mask.sum()
        anomaly_rate = (anomaly_count / total_shipments) * 100
        
        metrics_text = f"""
        📊 КЛЮЧЕВЫЕ МЕТРИКИ
        ─────────────────────
        Всего перевозок: {total_shipments:,}
        Аномальных маршрутов: {anomaly_count}
        Доля аномалий: {anomaly_rate:.1f}%
        
        Средний detour ratio: {self.data['detour_ratio'].mean():.2f}
        Средняя скорость: {self.data['avg_speed_kph'].mean():.1f} км/ч
        Средняя стоимость: {self.data['cost_rub'].mean():,.0f} руб
        
        ⚠️ ТОП-3 города по аномалиям:
        """
        
        city_anomalies = self.data[anomaly_mask].groupby('origin_city').size().nlargest(3)
        for city, count in city_anomalies.items():
            metrics_text += f"\n        • {city}: {count} аномалий"
        
        fig.text(0.02, 0.5, metrics_text, fontsize=11, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Добавляем графики
        ax1 = plt.axes([0.35, 0.55, 0.6, 0.4])
        ax1.hist(self.data['detour_ratio'][~anomaly_mask], bins=40, alpha=0.7, 
                label='Нормальные', color='green')
        ax1.hist(self.data['detour_ratio'][anomaly_mask], bins=15, alpha=0.7, 
                label='Аномальные', color='red')
        ax1.set_xlabel('Коэффициент отклонения')
        ax1.set_ylabel('Частота')
        ax1.set_title('Распределение отклонений маршрутов')
        ax1.legend()
        
        ax2 = plt.axes([0.35, 0.1, 0.25, 0.35])
        anomaly_by_day = self.data.groupby('day_of_week')['is_anomaly'].mean() * 100
        anomaly_by_day.plot(kind='bar', ax=ax2, color='coral')
        ax2.set_ylabel('Доля аномалий (%)')
        ax2.set_title('Аномалии по дням недели')
        ax2.tick_params(axis='x', rotation=45)
        
        ax3 = plt.axes([0.7, 0.1, 0.25, 0.35])
        speed_by_city = self.data.groupby('origin_city')['avg_speed_kph'].mean().nlargest(8)
        speed_by_city.plot(kind='barh', ax=ax3, color='lightgreen')
        ax3.set_xlabel('Средняя скорость (км/ч)')
        ax3.set_title('Скорость по городам')
        
        plt.suptitle('🚚 ДАШБОРД ЛОГИСТИЧЕСКИХ АНОМАЛИЙ', fontsize=16, fontweight='bold')
        plt.savefig(FIGURES_DIR / 'dashboard.png', dpi=150, bbox_inches='tight')
        plt.show()
