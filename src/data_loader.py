"""
Модуль для загрузки и предобработки данных грузоперевозок
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import Config, RAW_DATA_DIR, PROCESSED_DATA_DIR


class DataLoader:
    """Класс для работы с данными грузоперевозок"""
    
    def __init__(self):
        self.config = Config()
        self.data = None
        
    def generate_sample_data(self, n_rows=1000, save=True):
        """
        Генерация примера данных для демонстрации
        
        Parameters:
        -----------
        n_rows : int
            Количество записей
        save : bool
            Сохранить ли в CSV
        """
        np.random.seed(42)
        
        # Города и расстояния между ними
        cities = ['Москва', 'СПб', 'Казань', 'Новосибирск', 'Екатеринбург', 
                  'Нижний Новгород', 'Ростов-на-Дону', 'Самара', 'Красноярск']
        
        distances = {
            ('Москва', 'СПб'): 700, ('Москва', 'Казань'): 820,
            ('Москва', 'Екатеринбург'): 1800, ('Москва', 'Нижний Новгород'): 410,
            ('СПб', 'Казань'): 1500, ('Екатеринбург', 'Новосибирск'): 1600,
        }
        
        data = []
        
        for _ in range(n_rows):
            # Случайный выбор маршрута
            origin = np.random.choice(cities)
            destination = np.random.choice([c for c in cities if c != origin])
            
            # Оптимальное расстояние
            optimal_dist = distances.get((origin, destination), 
                                        distances.get((destination, origin), 
                                                    np.random.randint(200, 2000)))
            
            # Коэффициент отклонения (нормальные маршруты + аномалии)
            detour_ratio = np.random.normal(1.05, 0.08)
            # Добавляем аномалии (5% маршрутов с большим отклонением)
            if np.random.random() < 0.05:
                detour_ratio = np.random.uniform(1.6, 2.5)
            
            actual_distance = optimal_dist * detour_ratio
            
            # Время в пути (часы)
            # Нормальная скорость ~60 км/ч, аномалии ~20 км/ч или ~110 км/ч
            if detour_ratio > 1.5:
                avg_speed = np.random.choice([25, 110], p=[0.7, 0.3])
            else:
                avg_speed = np.random.normal(60, 10)
            
            travel_time_hours = actual_distance / avg_speed
            
            # Добавляем шум
            travel_time_hours += np.random.normal(0, 0.5)
            travel_time_hours = max(0.5, travel_time_hours)
            
            # Стоимость перевозки (руб)
            cost = actual_distance * 45 + np.random.normal(0, 5000)
            cost = max(1000, cost)
            
            data.append({
                'shipment_id': f'SHIP_{_:05d}',
                'origin_city': origin,
                'destination_city': destination,
                'optimal_distance_km': round(optimal_dist, 1),
                'actual_distance_km': round(actual_distance, 1),
                'travel_time_hours': round(travel_time_hours, 2),
                'cost_rub': round(cost, 2),
                'avg_speed_kph': round(actual_distance / travel_time_hours, 1),
                'detour_ratio': round(detour_ratio, 3),
                'date': pd.date_range('2024-01-01', periods=n_rows, freq='D')[_]
            })
        
        self.data = pd.DataFrame(data)
        
        if save:
            file_path = RAW_DATA_DIR / "shipments.csv"
            self.data.to_csv(file_path, index=False)
            print(f"✅ Данные сохранены в {file_path}")
            
        return self.data
    
    def load_data(self, file_path=None):
        """Загрузка данных из CSV"""
        if file_path is None:
            file_path = RAW_DATA_DIR / "shipments.csv"
        
        self.data = pd.read_csv(file_path, parse_dates=['date'])
        print(f"📊 Загружено {len(self.data)} записей")
        return self.data
    
    def preprocess_data(self):
        """Предобработка данных"""
        if self.data is None:
            raise ValueError("Данные не загружены. Сначала вызовите load_data()")
        
        df = self.data.copy()
        
        # Проверка пропусков
        print(f"Пропуски до обработки:\n{df.isnull().sum()}")
        df = df.dropna()
        
        # Фильтрация выбросов по скорости (явно невозможные значения)
        df = df[(df['avg_speed_kph'] > self.config.MIN_SPEED_KPH) & 
                (df['avg_speed_kph'] < self.config.MAX_SPEED_KPH)]
        
        # Добавление дополнительных признаков
        df['day_of_week'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
        
        # Эффективность маршрута (обратная detour_ratio)
        df['route_efficiency'] = 1 / df['detour_ratio']
        
        # Сохранение обработанных данных
        processed_path = PROCESSED_DATA_DIR / "processed_shipments.csv"
        df.to_csv(processed_path, index=False)
        
        self.data = df
        print(f"✅ Данные обработаны. Сохранено {len(df)} записей")
        
        return df
