import 'dart:developer';

import 'package:flutter/material.dart';
import 'package:mobile_scanner/mobile_scanner.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'home_screen.dart';

class QrScanScreen extends StatefulWidget {
  final String email;
  const QrScanScreen({super.key, required this.email});

  @override
  State<QrScanScreen> createState() => _QrScanScreenState();
}

class _QrScanScreenState extends State<QrScanScreen> {
  bool _scanned = false;
  String? _errorMessage;

  Future<void> _confirmPlate(String plate) async {
    setState(() {
      _scanned = true;
      _errorMessage = null;
    });
    try {
      final response = await http.post(
        Uri.parse('http://100.84.28.28:8000/confirm-plate'),
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: {
          'email': widget.email,
          'car_plat': plate,
        },
      );
      final data = jsonDecode(response.body);
      log('Response: ${data.toString()}');
      if (response.statusCode == 200 &&
          data['message'] == 'Plate linked and car entry registered') {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => HomeScreen(carNumber: plate)),
        );
      } else {
        setState(() {
          _errorMessage = data['detail'] ?? 'فشل الربط، تحقق من البيانات';
          _scanned = false;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'فشل الاتصال بالخادم';
        _scanned = false;
      });
    }
  }

  void _onDetect(BarcodeCapture capture) {
    if (_scanned) return;
    final Barcode? barcode = capture.barcodes.firstOrNull;
    if (barcode != null && barcode.rawValue != null) {
      _confirmPlate(barcode.rawValue!);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('فحص QR')),
      body: Stack(
        children: [
          MobileScanner(
            onDetect: _onDetect,
          ),
          if (_scanned) const Center(child: CircularProgressIndicator()),
          if (_errorMessage != null)
            Center(
              child: Container(
                padding: const EdgeInsets.all(24),
                margin: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: Colors.red.shade200),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(_errorMessage!,
                        style: const TextStyle(
                            color: Colors.red, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 16),
                    ElevatedButton(
                      onPressed: () => setState(() => _errorMessage = null),
                      child: const Text('إعادة المحاولة'),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
