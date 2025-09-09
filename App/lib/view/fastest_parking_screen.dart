import 'dart:developer';

import 'package:flutter/material.dart';
import 'dart:async';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'home_screen.dart';
import 'auth_screen.dart';

class FastestParkingScreen extends StatefulWidget {
  final String carNumber;
  const FastestParkingScreen({super.key, required this.carNumber});

  @override
  State<FastestParkingScreen> createState() => _FastestParkingScreenState();
}

class _FastestParkingScreenState extends State<FastestParkingScreen>
    with TickerProviderStateMixin {
  late AnimationController _pulseController;
  late AnimationController _slideController;
  late Animation<double> _pulseAnimation;
  late Animation<Offset> _slideAnimation;

  String? _fastestZone; // المنطقة ذات أكبر عدد أماكن فارغة
  int? _emptySpots;
  int? _totalSpots;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    )..repeat(reverse: true);

    _slideController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );

    _pulseAnimation = Tween<double>(begin: 0.8, end: 1.2).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 0.3),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _slideController,
      curve: Curves.elasticOut,
    ));

    _fetchFastestZone();
    _slideController.forward();
  }

  Future<void> _fetchFastestZone() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final response =
          await http.get(Uri.parse('http://100.104.75.37:8000/last-status'));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        // ابحث عن المنطقة ذات أكبر عدد empty
        String? maxZone;
        int maxEmpty = -1;
        int? total;
        for (final zone in ['A', 'B', 'C', 'D']) {
          final empty = data[zone.toLowerCase()]['empty'] ?? 0;
          if (empty > maxEmpty) {
            maxEmpty = empty;
            maxZone = zone;
            total = data[zone.toLowerCase()]['total'] ?? 0;
          }
        }
        setState(() {
          _fastestZone = maxZone;
          _emptySpots = maxEmpty;
          _totalSpots = total;
          _loading = false;
        });
      } else {
        setState(() {
          _error = 'فشل في جلب البيانات';
          _loading = false;
        });
      }
    } catch (e) {
      setState(() {
        _error = 'خطأ في الاتصال بالخادم';
        _loading = false;
      });
    }
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _slideController.dispose();
    super.dispose();
  }

  Future<void> _bookFastestSpot(BuildContext context) async {
    if (_fastestZone == null) return;
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const Center(child: CircularProgressIndicator()),
    );
    try {
      final response = await http.post(
        Uri.parse('http://100.84.28.28:8000/choose-segment'),
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: {
          'car_plat': widget.carNumber,
          'segment_id': _fastestZone!,
        },
      );
      Navigator.of(context).pop();
      final data = response.body.isNotEmpty ? jsonDecode(response.body) : {};
      if (response.statusCode == 200 && data['message'] == 'Segment updated') {
        final parentContext = Navigator.of(context).overlay?.context ?? context;
        if (parentContext.mounted) {
          showDialog(
            context: parentContext,
            barrierDismissible: false,
            builder: (ctx) => BookingSuccessDialog(
              zone: _fastestZone!,
              carNumber: widget.carNumber,
            ),
          );
        }
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(data['detail'] ?? 'فشل الحجز!')),
        );
      }
    } catch (e) {
      Navigator.of(context).pop();
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('فشل الاتصال بالخادم!')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Directionality(
      textDirection: TextDirection.rtl,
      child: Scaffold(
        backgroundColor: const Color(0xFFF8FAFC),
        body: _loading
            ? const Center(child: CircularProgressIndicator())
            : _error != null
                ? Center(
                    child: Text(_error!,
                        style: const TextStyle(color: Colors.red)))
                : CustomScrollView(
                    slivers: [
                      SliverAppBar(
                        expandedHeight: 120,
                        floating: false,
                        pinned: true,
                        elevation: 0,
                        backgroundColor: Colors.transparent,
                        leading: IconButton(
                          icon:
                              const Icon(Icons.arrow_back, color: Colors.white),
                          onPressed: () => Navigator.pop(context),
                        ),
                        flexibleSpace: Container(
                          decoration: const BoxDecoration(
                            gradient: LinearGradient(
                              colors: [Color(0xFF10B981), Color(0xFF059669)],
                              begin: Alignment.topRight,
                              end: Alignment.bottomLeft,
                            ),
                            borderRadius: BorderRadius.only(
                              bottomLeft: Radius.circular(30),
                              bottomRight: Radius.circular(30),
                            ),
                          ),
                          child: const FlexibleSpaceBar(
                            title: Text(
                              'أسرع موقف متاح',
                              style: TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                      ),
                      SliverToBoxAdapter(
                        child: SlideTransition(
                          position: _slideAnimation,
                          child: Padding(
                            padding: const EdgeInsets.all(24.0),
                            child: Column(
                              children: [
                                const SizedBox(height: 40),
                                Container(
                                  width: double.infinity,
                                  padding: const EdgeInsets.all(32),
                                  decoration: BoxDecoration(
                                    color: Colors.white,
                                    borderRadius: BorderRadius.circular(30),
                                    boxShadow: [
                                      BoxShadow(
                                        color: Colors.black.withOpacity(0.1),
                                        blurRadius: 20,
                                        offset: const Offset(0, 10),
                                      ),
                                    ],
                                  ),
                                  child: Column(
                                    children: [
                                      AnimatedBuilder(
                                        animation: _pulseAnimation,
                                        builder: (context, child) {
                                          return Transform.scale(
                                            scale: _pulseAnimation.value,
                                            child: Container(
                                              padding: const EdgeInsets.all(20),
                                              decoration: BoxDecoration(
                                                gradient: const LinearGradient(
                                                  colors: [
                                                    Color(0xFF10B981),
                                                    Color(0xFF059669)
                                                  ],
                                                ),
                                                shape: BoxShape.circle,
                                                boxShadow: [
                                                  BoxShadow(
                                                    color:
                                                        const Color(0xFF10B981)
                                                            .withOpacity(0.3),
                                                    blurRadius: 20,
                                                    spreadRadius: 5,
                                                  ),
                                                ],
                                              ),
                                              child: const Icon(
                                                Icons.local_parking,
                                                color: Colors.white,
                                                size: 60,
                                              ),
                                            ),
                                          );
                                        },
                                      ),
                                      const SizedBox(height: 32),
                                      const SizedBox(height: 16),
                                      Container(
                                        padding: const EdgeInsets.symmetric(
                                          horizontal: 24,
                                          vertical: 12,
                                        ),
                                        decoration: BoxDecoration(
                                          gradient: const LinearGradient(
                                            colors: [
                                              Color(0xFF10B981),
                                              Color(0xFF059669)
                                            ],
                                          ),
                                          borderRadius:
                                              BorderRadius.circular(25),
                                        ),
                                        child: Text(
                                          _fastestZone != null
                                              ? 'المنطقة $_fastestZone'
                                              : 'المنطقة الأسرع',
                                          style: const TextStyle(
                                            fontSize: 24,
                                            fontWeight: FontWeight.bold,
                                            color: Colors.white,
                                          ),
                                        ),
                                      ),
                                      const SizedBox(height: 24),
                                      Container(
                                        padding: const EdgeInsets.all(20),
                                        decoration: BoxDecoration(
                                          color: const Color(0xFF10B981)
                                              .withOpacity(0.1),
                                          borderRadius:
                                              BorderRadius.circular(20),
                                        ),
                                        child: Column(
                                          children: [
                                            _buildInfoRow(
                                              Icons.check_circle,
                                              'التوفر',
                                              _emptySpots != null &&
                                                      _totalSpots != null
                                                  ? '$_emptySpots / $_totalSpots متاح'
                                                  : '--',
                                            ),
                                            // يمكن إضافة معلومات أخرى هنا إذا توفرت
                                          ],
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                                const SizedBox(height: 32),
                                Row(
                                  children: [
                                    Expanded(
                                      child: _buildActionButton(
                                        'عرض الخريطة ',
                                        Icons.payment,
                                        const Color(0xFF10B981),
                                        () async {
                                          // استدعاء دالة الدفع من الديالوج
                                          final scaffold =
                                              ScaffoldMessenger.of(context);
                                          showDialog(
                                            context: context,
                                            barrierDismissible: false,
                                            builder: (ctx) => const Center(
                                                child:
                                                    CircularProgressIndicator()),
                                          );
                                          try {
                                            final response = await http.post(
                                              Uri.parse(
                                                  'http://100.84.28.28:8000/confirm-payment'),
                                              headers: {
                                                'Content-Type':
                                                    'application/x-www-form-urlencoded'
                                              },
                                              body: {
                                                'car_plat': widget.carNumber
                                              },
                                            );
                                            log(response.body);
                                            Navigator.of(context).pop();
                                            final body = response.body;
                                            bool paymentSuccess = false;
                                            if (response.statusCode == 200) {
                                              paymentSuccess = true;
                                            } else if (response.statusCode ==
                                                    404 &&
                                                body.contains(
                                                    'No unpaid booking found')) {
                                              paymentSuccess = true;
                                            } else {
                                              // تحقق من وجود رسالة نجاح في body
                                              try {
                                                final decoded =
                                                    jsonDecode(body);
                                                if (decoded is Map &&
                                                    decoded['detail'] ==
                                                        'Payment confirmed successfully.') {
                                                  paymentSuccess = true;
                                                }
                                              } catch (_) {}
                                            }
                                            if (paymentSuccess) {
                                              showDialog(
                                                context: context,
                                                barrierDismissible: false,
                                                builder: (ctx) =>
                                                    Directionality(
                                                  textDirection:
                                                      TextDirection.rtl,
                                                  child: AlertDialog(
                                                    shape:
                                                        RoundedRectangleBorder(
                                                            borderRadius:
                                                                BorderRadius
                                                                    .circular(
                                                                        20)),
                                                    title: const Text(
                                                        'تم الدفع بنجاح!',
                                                        textAlign:
                                                            TextAlign.center,
                                                        style: TextStyle(
                                                            fontWeight:
                                                                FontWeight
                                                                    .bold)),
                                                    content: const Text(
                                                        'تمت عملية الدفع بنجاح. يمكنك الآن بدء حجز جديد.'),
                                                    actionsAlignment:
                                                        MainAxisAlignment
                                                            .center,
                                                    actions: [
                                                      ElevatedButton(
                                                        style: ElevatedButton
                                                            .styleFrom(
                                                          backgroundColor:
                                                              const Color(
                                                                  0xFF10B981),
                                                          shape: RoundedRectangleBorder(
                                                              borderRadius:
                                                                  BorderRadius
                                                                      .circular(
                                                                          12)),
                                                        ),
                                                        onPressed: () {
                                                          Navigator.of(context)
                                                              .pushAndRemoveUntil(
                                                            MaterialPageRoute(
                                                                builder: (_) =>
                                                                    AuthScreen()),
                                                            (route) => false,
                                                          );
                                                        },
                                                        child: const Text(
                                                            'بداية حجز جديد',
                                                            style: TextStyle(
                                                                color: Colors
                                                                    .white)),
                                                      ),
                                                    ],
                                                  ),
                                                ),
                                              );
                                            } else {
                                              scaffold.showSnackBar(
                                                  const SnackBar(
                                                      content:
                                                          Text('فشل الدفع!')));
                                            }
                                          } catch (e) {
                                            Navigator.of(context).pop();
                                            scaffold.showSnackBar(const SnackBar(
                                                content: Text(
                                                    'فشل الاتصال بالخادم!')));
                                          }
                                        },
                                      ),
                                    ),
                                    const SizedBox(width: 16),
                                    Expanded(
                                      child: _buildActionButton(
                                        'استعلام الآن',
                                        Icons.search,
                                        const Color(0xFF273c75),
                                        () => _bookFastestSpot(context),
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 20),
                                SizedBox(
                                  width: double.infinity,
                                  height: 60,
                                  child: OutlinedButton.icon(
                                    onPressed: () => Navigator.pop(context),
                                    icon: const Icon(Icons.arrow_back),
                                    label: const Text(
                                      'العودة للبحث',
                                      style: TextStyle(fontSize: 16),
                                    ),
                                    style: OutlinedButton.styleFrom(
                                      shape: RoundedRectangleBorder(
                                        borderRadius: BorderRadius.circular(15),
                                      ),
                                      side: const BorderSide(
                                        color: Color(0xFF6B7280),
                                        width: 2,
                                      ),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
      ),
    );
  }

  Widget _buildInfoRow(IconData icon, String label, String value) {
    return Row(
      children: [
        Icon(
          icon,
          color: const Color(0xFF10B981),
          size: 24,
        ),
        const SizedBox(width: 12),
        Text(
          label,
          style: const TextStyle(
            fontSize: 16,
            color: Color(0xFF6B7280),
          ),
        ),
        const Spacer(),
        Text(
          value,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: Color(0xFF1F2937),
          ),
        ),
      ],
    );
  }

  Widget _buildActionButton(
    String text,
    IconData icon,
    Color color,
    VoidCallback onPressed,
  ) {
    return Container(
      height: 55,
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(15),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.3),
            blurRadius: 10,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(15),
          onTap: onPressed,
          child: Center(
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(icon, color: Colors.white, size: 20),
                const SizedBox(width: 8),
                Text(
                  text,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _BookingDialog extends StatefulWidget {
  final String carNumber;
  const _BookingDialog({Key? key, required this.carNumber}) : super(key: key);

  @override
  State<_BookingDialog> createState() => _BookingDialogState();
}

class _BookingDialogState extends State<_BookingDialog> {
  double cost = 0.0;
  int seconds = 0;
  Timer? timer;
  bool isStopped = false;
  bool _shouldShowSnackBar = false;

  @override
  void initState() {
    super.initState();
    timer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (!isStopped) {
        setState(() {
          seconds++;
          cost = (seconds / 3600) * 10; // 10 ريال لكل ساعة
        });
      }
    });
  }

  @override
  void dispose() {
    timer?.cancel();
    super.dispose();
  }

  void _stopBooking() {
    setState(() {
      isStopped = true;
      _shouldShowSnackBar = true;
    });
    // بعد إغلاق الديالوج المؤقت، اعرض ديالوج معلومات الحجز
    Future.delayed(const Duration(milliseconds: 300), () {
      if (mounted && _shouldShowSnackBar) {
        Navigator.of(context).pop();
        // بعد إغلاق الديالوج المؤقت، اعرض ديالوج معلومات الحجز
        Future.delayed(const Duration(milliseconds: 200), () {
          final parentContext = Navigator.of(context).overlay?.context;
          if (parentContext != null) {
            showDialog(
              context: parentContext,
              barrierDismissible: false,
              builder: (ctx) => BookingSuccessDialog(
                zone: 'ب', // المنطقة الافتراضية هنا (يمكنك تعديلها حسب منطقك)
                carNumber: widget.carNumber,
              ),
            );
          }
        });
      }
    });
  }

  void _closeDialog() {
    timer?.cancel();
    Navigator.pop(context);
  }

  Future<void> _confirmPayment() async {
    final scaffold = ScaffoldMessenger.of(context);
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => const Center(child: CircularProgressIndicator()),
    );
    try {
      final response = await http.post(
        Uri.parse('http://100.84.28.28:8000/confirm-payment'),
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: {'car_plat': widget.carNumber},
      );
      Navigator.of(context).pop();
      if (response.statusCode == 200) {
        scaffold.showSnackBar(const SnackBar(content: Text('تم الدفع بنجاح!')));
        Navigator.of(context).pop();
      } else if (response.statusCode == 404 &&
          response.body.contains('No unpaid booking found')) {
        scaffold.showSnackBar(const SnackBar(content: Text('تم الدفع بنجاح!')));
        Navigator.of(context).pop();
      } else {
        scaffold.showSnackBar(const SnackBar(content: Text('فشل الدفع!')));
      }
    } catch (e) {
      Navigator.of(context).pop();
      scaffold
          .showSnackBar(const SnackBar(content: Text('فشل الاتصال بالخادم!')));
    }
  }

  String _formatDuration(int totalSeconds) {
    final minutes = (totalSeconds ~/ 60).toString().padLeft(2, '0');
    final secs = (totalSeconds % 60).toString().padLeft(2, '0');
    return '$minutes:$secs';
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      backgroundColor: Colors.white,
      child: Padding(
        padding: const EdgeInsets.all(28.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.bookmark, color: Color(0xFFEF4444), size: 48),
            const SizedBox(height: 16),
            const Text(
              'الحجز قيد التنفيذ',
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: Color(0xFF1F2937),
              ),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
              decoration: BoxDecoration(
                color: const Color(0xFFEF4444).withOpacity(0.08),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.timer, color: Color(0xFFEF4444)),
                      const SizedBox(width: 8),
                      Text(
                        _formatDuration(seconds),
                        style: const TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.bold,
                          color: Color(0xFFEF4444),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.attach_money, color: Color(0xFFEF4444)),
                      const SizedBox(width: 8),
                      Text(
                        '${cost.toStringAsFixed(2)} ريال',
                        style: const TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          color: Color(0xFFEF4444),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor:
                          isStopped ? Colors.grey : const Color(0xFFEF4444),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    onPressed: isStopped ? null : _stopBooking,
                    icon: const Icon(Icons.stop_circle, color: Colors.white),
                    label: const Text(
                      'إيقاف العداد',
                      style: TextStyle(fontSize: 16, color: Colors.white),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF10B981),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    onPressed: _confirmPayment,
                    icon: const Icon(Icons.payment, color: Colors.white),
                    label: const Text(
                      'دفع الرسوم',
                      style: TextStyle(fontSize: 16, color: Colors.white),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                IconButton(
                  icon: const Icon(Icons.close,
                      color: Color(0xFF6B7280), size: 28),
                  onPressed: _closeDialog,
                  tooltip: 'إغلاق',
                ),
              ],
            ),
            if (isStopped) ...[
              const SizedBox(height: 12),
              const Text(
                'تم إيقاف العداد. يمكنك الآن معرفة التكلفة النهائية.',
                style: TextStyle(fontSize: 14, color: Color(0xFF6B7280)),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
