
import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt


sys.path.append(r'C:\Users\Evgeniy\Desktop\logistics_analysis')

warnings.filterwarnings('ignore')

from src.data_loader import DataLoader
from src.anomaly_detector import AnomalyDetector
from src.visualizer import Visualizer
from src.database import LogisticsDatabase


def main():
    print("=" * 60)
    print("АНАЛИЗ ЛОГИСТИЧЕСКИХ ДАННЫХ И ВЫЯВЛЕНИЕ АНОМАЛИЙ")
    print("=" * 60)

    print("\nШаг 1: Загрузка данных...")
    loader = DataLoader()


    data_raw = loader.generate_sample_data(n_rows=2000)
    print(f"   Сгенерировано {len(data_raw)} записей")

    data = loader.load_data()
    data = loader.preprocess_data()

    print(f"\nСтатистика данных:")
    print(f"   - Всего записей: {len(data)}")
    print(f"   - Уникальных маршрутов: {data.groupby(['origin_city', 'destination_city']).ngroups}")
    print(f"   - Период: {data['date'].min().date()} - {data['date'].max().date()}")

    print("\nШаг 2: Выявление аномальных маршрутов...")
    detector = AnomalyDetector(data)

   
    anomalies = detector.detect_by_statistical_methods()

   
    iso_anomalies = detector.detect_by_isolation_forest(contamination=0.05)

    
    anomaly_summary = detector.get_anomaly_summary()
    print("\nРезультаты обнаружения аномалий:")
    print(anomaly_summary.to_string())

   
    analysis_results = detector.analyze_anomalies_by_city()
    print(f"\nОбщая статистика:")
    print(f"   - Всего аномалий: {analysis_results['total_anomalies']}")
    print(f"   - Доля аномалий: {analysis_results['anomaly_percentage']:.2f}%")

    print("\nСтатистика по городам (ТОП-5 по аномалиям):")
    city_stats = analysis_results['city_statistics'].nlargest(5, 'anomaly_rate')
    print(city_stats.to_string())

    print("\n10 самых проблемных маршрутов:")
    print(analysis_results['worst_routes'].to_string())

  
    print("\nШаг 3: Создание визуализаций...")
    data['is_anomaly'] = anomalies['combined']
    visualizer = Visualizer(data)

 
    visualizer.plot_anomaly_distribution(anomalies['combined'])
    visualizer.plot_route_efficiency()
    visualizer.plot_cost_analysis()
    visualizer.create_dashboard(anomalies['combined'])

    print("   Графики сохранены в папке reports/figures/")

   
    print("\nШаг 4: Сохранение в базу данных...")
    db = LogisticsDatabase()
    db.create_tables()
    db.insert_shipments(data)
    db.update_anomalies(anomalies['combined'], 'combined')
    db.update_route_statistics(data)

  
    top_routes = db.get_top_anomalous_routes(limit=10)
    print("\nТОП-10 самых аномальных маршрутов (из БД):")
    print(top_routes.to_string())

    detailed_report = db.get_detailed_anomaly_report()
    print(f"\nДетальный отчет: найдено {len(detailed_report)} аномалий")
    db.close()

    
    print("\n" + "=" * 60)
    print(" ИТОГОВЫЙ ОТЧЕТ ПО АНАЛИЗУ")
    print("=" * 60)

    
    anomaly_by_day = data.groupby('day_of_week')['is_anomaly'].mean() * 100
    print("\nАномалии по дням недели (%):")
    for day, rate in anomaly_by_day.items():
        print(f"   {day}: {rate:.1f}%")

    
    city_efficiency = data.groupby('origin_city')['route_efficiency'].mean()
    best_city = city_efficiency.idxmax()
    worst_city = city_efficiency.idxmin()

    print(f"\nСамый эффективный город: {best_city} (эффективность: {city_efficiency[best_city]:.2f})")
    print(f"Самый неэффективный город: {worst_city} (эффективность: {city_efficiency[worst_city]:.2f})")

    
    anomaly_cost = data[data['is_anomaly']]['cost_rub'].sum()
    normal_avg_cost = data[~data['is_anomaly']]['cost_rub'].mean()
    potential_savings = anomaly_cost - (len(data[data['is_anomaly']]) * normal_avg_cost)

    print(f"\nПотенциальная экономия при устранении аномалий: {potential_savings:,.0f} руб")

    print("\n" + "=" * 60)
    print("АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
    print("Результаты сохранены в:")
    print("   - reports/figures/ - графики")
    print("   - data/processed/ - обработанные данные")
    print("   - logistics.db - база данных")
    print("=" * 60)

   
    plt.show(block=True)


if __name__ == "__main__":
    main()
