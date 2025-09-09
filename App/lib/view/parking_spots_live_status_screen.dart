import 'package:flutter/material.dart';

class ParkingSpotsLiveStatusScreen extends StatelessWidget {
  final Map<String, dynamic> spotsData;
  const ParkingSpotsLiveStatusScreen({Key? key, required this.spotsData})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('مواقف السيارات المتاحة'),
        backgroundColor: Colors.deepPurple,
        centerTitle: true,
      ),
      backgroundColor: const Color(0xFFF8FAFC),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'الحالة الحية للمواقف',
              style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF1F2937)),
            ),
            const SizedBox(height: 24),
            Expanded(
              child: ListView.separated(
                itemCount: spotsData.length,
                separatorBuilder: (context, i) => const SizedBox(height: 16),
                itemBuilder: (context, i) {
                  final zone = spotsData.keys.elementAt(i);
                  final data = spotsData[zone];
                  return Card(
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(18)),
                    elevation: 4,
                    child: ListTile(
                      leading: Icon(Icons.local_parking,
                          color: Colors.green, size: 36),
                      title: Text('منطقة $zone',
                          style: const TextStyle(
                              fontWeight: FontWeight.bold, fontSize: 18)),
                      subtitle: Row(
                        children: [
                          Icon(
                            data['available'] > 0
                                ? Icons.check_circle
                                : Icons.cancel,
                            color: data['available'] > 0
                                ? Colors.green
                                : Colors.red,
                            size: 20,
                          ),
                          const SizedBox(width: 8),
                          Text(
                            data['available'] > 0
                                ? '${data['available']} موقف متاح'
                                : 'لا يوجد أماكن متاحة',
                            style: TextStyle(
                              color: data['available'] > 0
                                  ? Colors.green
                                  : Colors.red,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          const SizedBox(width: 16),
                          const Icon(Icons.directions_car,
                              color: Colors.blueGrey, size: 18),
                          const SizedBox(width: 4),
                          Text('إجمالي: ${data['total']}'),
                        ],
                      ),
                      trailing: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(
                          color: data['available'] > 0
                              ? Colors.green[50]
                              : Colors.red[50],
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          data['available'] > 0 ? 'متاح' : 'مغلق',
                          style: TextStyle(
                            color: data['available'] > 0
                                ? Colors.green
                                : Colors.red,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// مثال على الاستخدام:
// ParkingSpotsLiveStatusScreen(
//   spotsData: {
//     'A': {'available': 5, 'total': 20},
//     'B': {'available': 0, 'total': 15},
//     'C': {'available': 3, 'total': 10},
//     'D': {'available': 7, 'total': 25},
//   },
// )
