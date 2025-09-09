import 'dart:convert';
import 'dart:developer';

import 'package:flutter/material.dart';
import 'dart:async';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import '../../global_timer_provider.dart';
import 'fastest_parking_screen.dart';
import 'auth_screen.dart';
import 'package:http/http.dart' as http;

class HomeScreen extends StatefulWidget {
  final String carNumber;
  const HomeScreen({super.key, required this.carNumber});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  // بيانات الأماكن المتاحة لكل منطقة
  Map<String, int> _zoneEmptySpots = {};
  Map<String, int> _zoneTotalSpots = {};
  bool _loadingZoneSpots = false;
  String? _zoneSpotsError;

  // حذف didChangeDependencies نهائياً، سيتم التحميل في initState فقط

  Future<void> _fetchZoneSpots() async {
    setState(() {
      _loadingZoneSpots = true;
      _zoneSpotsError = null;
    });
    try {
      final response =
          await http.get(Uri.parse('http://100.104.75.37:8000/last-status'));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _zoneEmptySpots = {
            'A': data['a']['empty'] ?? 0,
            'B': data['b']['empty'] ?? 0,
            'C': data['c']['empty'] ?? 0,
            'D': data['d']['empty'] ?? 0,
          };
          _zoneTotalSpots = {
            'A': data['a']['total'] ?? 0,
            'B': data['b']['total'] ?? 0,
            'C': data['c']['total'] ?? 0,
            'D': data['d']['total'] ?? 0,
          };
          _loadingZoneSpots = false;
        });
      } else {
        setState(() {
          _zoneSpotsError = 'فشل في جلب بيانات المناطق';
          _loadingZoneSpots = false;
        });
      }
    } catch (e) {
      setState(() {
        _zoneSpotsError = 'خطأ في الاتصال بالخادم';
        _loadingZoneSpots = false;
      });
    }
  }

  String? selectedPlace;
  List<String>? selectedZones;

  // Animation Controllers
  late AnimationController _mainAnimationController;
  late AnimationController _cardAnimationController;
  late AnimationController _pulseController;
  late AnimationController _floatingController;
  late AnimationController _glowController;

  // Animations
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;
  late Animation<Offset> _cardSlideAnimation;
  late Animation<double> _scaleAnimation;
  late Animation<double> _pulseAnimation;
  late Animation<double> _floatingAnimation;
  late Animation<double> _glowAnimation;
  late Animation<double> _rotationAnimation;

  final Map<String, dynamic> places = {
    'السينما': {
      'icon': Icons.local_movies,
      'color': const Color(0xFF4d194d),
      'zones': ['A', 'B', 'C', 'D'],
      'gradient': [const Color(0xFF4d194d), const Color(0xFF764BA2)],
    },
    'المطاعم': {
      'icon': Icons.restaurant,
      'color': const Color(0xFF012622),
      'zones': ['D', 'A', 'C', 'B'],
      'gradient': [const Color(0xFF012622), const Color(0xFF003b36)],
    },
    'المولات': {
      'icon': Icons.shopping_bag,
      'color': const Color(0xFF003249), // لون تركواز عصري
      'zones': ['C', 'D', 'B', 'A'],
      'gradient': [
        const Color(0xFF007ea7),
        const Color(0xFF003249)
      ], // تدرج تركواز فاتح
    },
    'المكاتب': {
      'icon': Icons.business,
      'color': const Color(0xFF38040e), // لون برتقالي ذهبي واضح
      'zones': ['B', 'D', 'C', 'A'],
      'gradient': [
        const Color(0xFF38040e),
        const Color(0xFF800e13)
      ], // تدرج ذهبي فاتح
    },
  };

  int _selectedIndex = 0;

  static final List<Widget> _pages = <Widget>[
    // الصفحة الرئيسية (سيتم استبدالها في build)
    const SizedBox.shrink(),
    _UserHistoryScreen(),
    // الملف الشخصي (سيتم استبداله في build)
    const SizedBox.shrink(),
    _SettingsScreen(),
  ];

  void _onNavBarTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  void initState() {
    super.initState();
    log('***************************************************************************************************************${widget.carNumber}');
    _initializeAnimations();
    _startAnimations();
    // تحميل بيانات الأماكن مرة واحدة فقط عند فتح الصفحة
    _fetchZoneSpots();
  }

  void _initializeAnimations() {
    // Main animation controller
    _mainAnimationController = AnimationController(
      duration: const Duration(milliseconds: 1200),
      vsync: this,
    );

    // Card animation controller
    _cardAnimationController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );

    // Pulse controller
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 2000),
      vsync: this,
    )..repeat(reverse: true);

    // Floating controller
    _floatingController = AnimationController(
      duration: const Duration(milliseconds: 3000),
      vsync: this,
    )..repeat(reverse: true);

    // Glow controller
    _glowController = AnimationController(
      duration: const Duration(milliseconds: 2500),
      vsync: this,
    )..repeat(reverse: true);

    // Initialize animations
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
          parent: _mainAnimationController, curve: Curves.easeOutQuart),
    );

    _slideAnimation = Tween<Offset>(
      begin: const Offset(1.0, 0.0), // Start from right for RTL
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _mainAnimationController,
      curve: Curves.elasticOut,
    ));

    _cardSlideAnimation = Tween<Offset>(
      begin: const Offset(0.0, 0.5),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _cardAnimationController,
      curve: Curves.easeOutBack,
    ));

    _scaleAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
          parent: _cardAnimationController, curve: Curves.elasticOut),
    );

    _pulseAnimation = Tween<double>(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    _floatingAnimation = Tween<double>(begin: -5.0, end: 5.0).animate(
      CurvedAnimation(parent: _floatingController, curve: Curves.easeInOut),
    );

    _glowAnimation = Tween<double>(begin: 0.3, end: 0.8).animate(
      CurvedAnimation(parent: _glowController, curve: Curves.easeInOut),
    );

    _rotationAnimation = Tween<double>(begin: 0.0, end: 0.1).animate(
      CurvedAnimation(parent: _floatingController, curve: Curves.easeInOut),
    );
  }

  void _startAnimations() {
    _mainAnimationController.forward();
  }

  @override
  void dispose() {
    _mainAnimationController.dispose();
    _cardAnimationController.dispose();
    _pulseController.dispose();
    _floatingController.dispose();
    _glowController.dispose();
    super.dispose();
  }

  Widget _buildAnimatedWelcomeCard() {
    return AnimatedBuilder(
      animation: _floatingAnimation,
      builder: (context, child) {
        return Transform.translate(
          offset: Offset(0, _floatingAnimation.value),
          child: Transform.rotate(
            angle: _rotationAnimation.value * 0.1,
            child: Container(
              padding: const EdgeInsets.all(28),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  begin: Alignment.topRight,
                  end: Alignment.bottomLeft,
                  colors: [
                    Color(0xFFFFFFFF),
                    Color(0xFFF8FAFC),
                  ],
                ),
                borderRadius: BorderRadius.circular(24),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.08),
                    blurRadius: 20,
                    offset: const Offset(0, 8),
                    spreadRadius: -4,
                  ),
                  BoxShadow(
                    color: const Color(0xFF667EEA).withOpacity(0.1),
                    blurRadius: 40,
                    offset: const Offset(0, 16),
                    spreadRadius: -8,
                  ),
                ],
              ),
              child: Row(
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
                              colors: [Color(0xFF667EEA), Color(0xFF764BA2)],
                            ),
                            borderRadius: BorderRadius.circular(20),
                            boxShadow: [
                              BoxShadow(
                                color: const Color(0xFF667EEA).withOpacity(0.4),
                                blurRadius: 16,
                                offset: const Offset(0, 8),
                              ),
                            ],
                          ),
                          child: const Icon(
                            Icons.waving_hand,
                            color: Colors.white,
                            size: 32,
                          ),
                        ),
                      );
                    },
                  ),
                  const SizedBox(width: 20),
                  const Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'أهلاً وسهلاً بك!',
                          style: TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: Color(0xFF1F2937),
                            letterSpacing: -0.5,
                          ),
                        ),
                        SizedBox(height: 8),
                        Text(
                          'اكتشف أفضل مواقف السيارات في مدينتك',
                          style: TextStyle(
                            fontSize: 16,
                            color: Color(0xFF0d3b66),
                            fontWeight: FontWeight.bold,
                            height: 1.4,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildEnhancedPlaceSelector() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 24,
            offset: const Offset(0, 12),
            spreadRadius: -6,
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: DropdownButtonFormField<String>(
          decoration: InputDecoration(
            labelText: 'اختر وجهتك المفضلة',
            labelStyle: const TextStyle(
              color: Color(0xFF6B7280),
              fontSize: 20,
              fontWeight: FontWeight.w600,
            ),
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(20),
              borderSide: BorderSide.none,
            ),
            filled: true,
            fillColor: Colors.white,
            prefixIcon: Container(
              margin: const EdgeInsets.all(12),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF667EEA), Color(0xFF764BA2)],
                ),
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFF667EEA).withOpacity(0.3),
                    blurRadius: 12,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: const Icon(
                Icons.place,
                color: Colors.white,
                size: 20,
              ),
            ),
            contentPadding: const EdgeInsets.symmetric(
              horizontal: 24,
              vertical: 20,
            ),
          ),
          value: selectedPlace,
          icon: const Icon(
            Icons.keyboard_arrow_down,
            color: Color(0xFF667EEA),
            size: 28,
          ),
          style: const TextStyle(
            fontSize: 16,
            color: Color(0xFF1F2937),
            fontWeight: FontWeight.w600,
          ),
          items: places.keys.map((place) {
            final placeData = places[place]!;
            return DropdownMenuItem<String>(
              value: place,
              child: Directionality(
                textDirection: TextDirection.rtl,
                child: Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: placeData['gradient'] as List<Color>,
                        ),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Icon(
                        placeData['icon'] as IconData,
                        color: Colors.white,
                        size: 20,
                      ),
                    ),
                    const SizedBox(width: 16),
                    Text(
                      place,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            );
          }).toList(),
          onChanged: (value) {
            if (value != null) {
              HapticFeedback.lightImpact();
            }
            setState(() {
              selectedPlace = value;
              selectedZones = value != null
                  ? List<String>.from(places[value]!['zones'])
                  : null;
            });
            if (value != null) {
              _cardAnimationController.reset();
              _cardAnimationController.forward();
            }
          },
        ),
      ),
    );
  }

  Widget _buildAnimatedZonesCard() {
    if (selectedPlace == null || selectedZones == null) {
      return const SizedBox.shrink();
    }
    final placeData = places[selectedPlace]!;
    // لا نعيد التحميل كل مرة، فقط نظهر البيانات مرة واحدة بعد التحميل
    if (_loadingZoneSpots) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_zoneSpotsError != null) {
      return Center(
          child: Text(_zoneSpotsError!,
              style: const TextStyle(color: Colors.red)));
    }
    return AnimatedBuilder(
      animation: _cardAnimationController,
      builder: (context, child) {
        return FadeTransition(
          opacity: _cardAnimationController,
          child: SlideTransition(
            position: _cardSlideAnimation,
            child: ScaleTransition(
              scale: _scaleAnimation,
              child: Container(
                padding: const EdgeInsets.all(28),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topRight,
                    end: Alignment.bottomLeft,
                    colors: [
                      (placeData['gradient'] as List<Color>)[0]
                          .withOpacity(0.1),
                      (placeData['gradient'] as List<Color>)[1]
                          .withOpacity(0.05),
                      Colors.white.withOpacity(0.9),
                    ],
                  ),
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(
                    color: (placeData['color'] as Color).withOpacity(0.2),
                    width: 1.5,
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: (placeData['color'] as Color).withOpacity(0.15),
                      blurRadius: 24,
                      offset: const Offset(0, 12),
                      spreadRadius: -4,
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        AnimatedBuilder(
                          animation: _glowAnimation,
                          builder: (context, child) {
                            return Container(
                              padding: const EdgeInsets.all(16),
                              decoration: BoxDecoration(
                                gradient: LinearGradient(
                                  colors: placeData['gradient'] as List<Color>,
                                ),
                                borderRadius: BorderRadius.circular(16),
                                boxShadow: [
                                  BoxShadow(
                                    color: (placeData['color'] as Color)
                                        .withOpacity(_glowAnimation.value
                                            .clamp(0.0, 1.0)),
                                    blurRadius: 20,
                                    offset: const Offset(0, 8),
                                  ),
                                ],
                              ),
                              child: Icon(
                                placeData['icon'] as IconData,
                                color: Colors.white,
                                size: 28,
                              ),
                            );
                          },
                        ),
                        const SizedBox(width: 20),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'المناطق المتاحة',
                                style: TextStyle(
                                  fontWeight: FontWeight.w800,
                                  color: placeData['color'] as Color,
                                  fontSize: 20,
                                  letterSpacing: -0.5,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                selectedPlace!,
                                style: TextStyle(
                                  color: (placeData['color'] as Color)
                                      .withOpacity(0.7),
                                  fontSize: 15,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 24),
                    Column(
                      children: selectedZones!.asMap().entries.map((entry) {
                        final index = entry.key;
                        final zone = entry.value;
                        final empty = _zoneEmptySpots[zone] ?? 0;
                        final total = _zoneTotalSpots[zone] ?? 0;
                        return TweenAnimationBuilder<double>(
                          duration: Duration(milliseconds: 400 + (index * 100)),
                          tween: Tween(begin: 0.0, end: 1.0),
                          curve: Curves.elasticOut,
                          builder: (context, value, child) {
                            return Transform.scale(
                              scale: value,
                              child: Transform.translate(
                                offset: Offset(0, (1 - value) * 20),
                                child: Opacity(
                                  opacity: value.clamp(0.0, 1.0),
                                  child: GestureDetector(
                                    onTap: () {
                                      HapticFeedback.lightImpact();
                                      _showZoneDetails(zone, placeData);
                                    },
                                    child: Container(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 20,
                                        vertical: 16,
                                      ),
                                      decoration: BoxDecoration(
                                        gradient: LinearGradient(
                                          colors: placeData['gradient']
                                              as List<Color>,
                                        ),
                                        borderRadius: BorderRadius.circular(20),
                                        boxShadow: [
                                          BoxShadow(
                                            color: (placeData['color'] as Color)
                                                .withOpacity(0.4),
                                            blurRadius: 12,
                                            offset: const Offset(0, 6),
                                          ),
                                        ],
                                      ),
                                      margin: const EdgeInsets.only(bottom: 12),
                                      child: SizedBox(
                                        width: double.infinity,
                                        child: Row(
                                          mainAxisSize: MainAxisSize.min,
                                          children: [
                                            const Icon(
                                              Icons.local_parking,
                                              color: Colors.white,
                                              size: 18,
                                            ),
                                            const SizedBox(width: 8),
                                            Text(
                                              'منطقة $zone',
                                              style: const TextStyle(
                                                color: Colors.white,
                                                fontWeight: FontWeight.w700,
                                                fontSize: 14,
                                              ),
                                            ),
                                            Spacer(),
                                            Row(
                                              children: [
                                                Icon(Icons.event_seat,
                                                    color: Colors.yellow[700],
                                                    size: 18),
                                                const SizedBox(width: 2),
                                                Text(
                                                  '$empty / $total متاح',
                                                  style: const TextStyle(
                                                    color: Colors.white,
                                                    fontWeight: FontWeight.bold,
                                                    fontSize: 13,
                                                  ),
                                                ),
                                              ],
                                            ),
                                          ],
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                            );
                          },
                        );
                      }).toList(),
                    ),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildEnhancedActionButton(
    String text,
    IconData icon,
    List<Color> gradient,
    VoidCallback onPressed,
  ) {
    return AnimatedBuilder(
      animation: _pulseAnimation,
      builder: (context, child) {
        return Transform.scale(
          scale: _pulseAnimation.value,
          child: Container(
            height: 65,
            decoration: BoxDecoration(
              gradient: LinearGradient(colors: gradient),
              borderRadius: BorderRadius.circular(20),
              boxShadow: [
                BoxShadow(
                  color: gradient[0].withOpacity(0.4),
                  blurRadius: 20,
                  offset: const Offset(0, 10),
                  spreadRadius: -4,
                ),
              ],
            ),
            child: Material(
              color: Colors.transparent,
              child: InkWell(
                borderRadius: BorderRadius.circular(20),
                onTap: () {
                  HapticFeedback.mediumImpact();
                  onPressed();
                },
                child: Center(
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(icon, color: Colors.white, size: 24),
                      const SizedBox(width: 12),
                      Text(
                        text,
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w700,
                          fontSize: 16,
                          letterSpacing: 0.5,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  void _showZoneDetails(String zone, Map<String, dynamic> placeData) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => Directionality(
        textDirection: TextDirection.rtl,
        child: _ZoneDetailsSheet(
            zone: zone,
            placeData: placeData,
            carNumber: widget.carNumber,
            totalSpots: _zoneTotalSpots[zone] ?? 0,
            emptySpots: _zoneEmptySpots[zone] ?? 0),
      ),
    );
  }

  void _showBookingDialog(BuildContext context) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const _EnhancedBookingDialog(),
    );
  }

  @override
  Widget build(BuildContext context) {
    final timerProvider = Provider.of<GlobalTimerProvider>(context);
    final List<Widget> pages = List.from(_pages);
    // الصفحة الرئيسية (Home) بمحتوى التطبيق
    pages[0] = Directionality(
      textDirection: TextDirection.rtl,
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildAnimatedWelcomeCard(),
            const SizedBox(height: 24),
            _buildEnhancedPlaceSelector(),
            const SizedBox(height: 24),
            _buildAnimatedZonesCard(),
            const SizedBox(height: 32),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                icon: const Icon(Icons.flash_on, color: Colors.white),
                label: const Text(
                  'أسرع موقف متاح الآن',
                  style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.white),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF10B981),
                  padding: const EdgeInsets.symmetric(vertical: 18),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(18),
                  ),
                  elevation: 6,
                ),
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) =>
                          FastestParkingScreen(carNumber: widget.carNumber),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
    // الملف الشخصي
    pages[2] = _UserProfileScreen(carNumber: widget.carNumber);
    return Directionality(
      textDirection: TextDirection.rtl,
      child: Scaffold(
        backgroundColor: const Color(0xFFF8FAFC),
        appBar: AppBar(
          backgroundColor: Colors.deepPurple,
          elevation: 0,
          flexibleSpace: Container(
            decoration: const BoxDecoration(
              color: Color(0xFF001f54),
            ),
          ),
          title: const Text(
            'مواقف السيارات الذكية',
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w800,
              fontSize: 22,
              letterSpacing: -0.5,
            ),
          ),
          centerTitle: true,
          actions: [
            IconButton(
              onPressed: () {},
              icon: const Icon(
                Icons.notifications_outlined,
                color: Color(0xFF6B7280),
              ),
            ),
          ],
        ),
        body: pages[_selectedIndex],
        bottomNavigationBar: BottomNavigationBar(
          currentIndex: _selectedIndex,
          onTap: _onNavBarTapped,
          selectedItemColor: Colors.orange,
          unselectedItemColor: Colors.white,
          backgroundColor: const Color(0xFF001f54),
          showUnselectedLabels: true,
          type: BottomNavigationBarType.fixed,
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.home),
              label: 'الرئيسية',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.history),
              label: 'التاريخ',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.person),
              label: 'الملف الشخصي',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.settings),
              label: 'الإعدادات',
            ),
          ],
        ),
      ),
    );
  }

  String _formatDuration(int seconds) {
    final d = Duration(seconds: seconds);
    String twoDigits(int n) => n.toString().padLeft(2, '0');
    final h = twoDigits(d.inHours);
    final m = twoDigits(d.inMinutes.remainder(60));
    final s = twoDigits(d.inSeconds.remainder(60));
    return h != '00' ? '$h:$m:$s' : '$m:$s';
  }
}

class _ZoneDetailsSheet extends StatelessWidget {
  final String zone;
  final Map<String, dynamic> placeData;
  final String carNumber;
  final int totalSpots;
  final int emptySpots;

  const _ZoneDetailsSheet({
    required this.zone,
    required this.placeData,
    required this.carNumber,
    required this.totalSpots,
    required this.emptySpots,
  });

  Future<void> _chooseSegment(BuildContext context, String segmentId,
      BuildContext parentContext) async {
    log('carNumber in ZoneDetailsSheet: $carNumber');
    if (carNumber.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('رقم السيارة غير متوفر!')));
      return;
    }

    // إظهار loading indicator
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const Center(
        child: CircularProgressIndicator(),
      ),
    );

    try {
      final response = await http.post(
        Uri.parse('http://100.84.28.28:8000/choose-segment'),
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: {
          'car_plat': carNumber,
          'segment_id': segmentId,
        },
      );

      // إخفاء loading indicator
      Navigator.of(context).pop();

      final data = jsonDecode(response.body);
      log(data.toString());

      if (response.statusCode == 200 && data['message'] == 'Segment updated') {
        // إغلاق الـ bottom sheet أولاً
        Navigator.of(context).pop();

        // انتظار قصير للتأكد من انتهاء animation الإغلاق
        await Future.delayed(const Duration(milliseconds: 300));

        // عرض dialog النجاح باستخدام parentContext (من مستوى أعلى)
        if (parentContext.mounted) {
          showDialog(
            context: parentContext,
            barrierDismissible: false,
            builder: (ctx) => BookingSuccessDialog(
              zone: segmentId,
              carNumber: carNumber,
            ),
          );
        }
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text(data['detail'] ?? 'فشل تحديث المنطقة!')));
      }
    } catch (e) {
      // إخفاء loading indicator في حالة الخطأ
      Navigator.of(context).pop();

      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text('فشل الاتصال بالخادم!')));

      log('Error in _chooseSegment: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 40,
            height: 4,
            decoration: BoxDecoration(
              color: Colors.grey[300],
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(height: 24),
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: placeData['gradient'] as List<Color>,
                  ),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: const Icon(
                  Icons.local_parking,
                  color: Colors.white,
                  size: 24,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'منطقة $zone',
                      style: const TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Color(0xFF1F2937),
                      ),
                    ),
                    Text(
                      '$emptySpots / $totalSpots موقف متاح',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey[600],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),
          Row(
            children: [
              Expanded(
                child: ElevatedButton(
                  onPressed: () {
                    // احصل على context من مستوى أعلى (مثلاً الـ overlay)
                    final parentContext =
                        Navigator.of(context).overlay?.context ?? context;
                    _chooseSegment(context, zone, parentContext);
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: placeData['color'],
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: const Text(
                    'احجز الآن',
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
        ],
      ),
    );
  }
}

// Dialog to show booking success and allow inquiry
// Dialog to show booking success and allow inquiry
class BookingSuccessDialog extends StatefulWidget {
  final String zone;
  final String carNumber;
  const BookingSuccessDialog(
      {Key? key, required this.zone, required this.carNumber})
      : super(key: key);

  @override
  State<BookingSuccessDialog> createState() => BookingSuccessDialogState();
}

class BookingSuccessDialogState extends State<BookingSuccessDialog> {
  bool loading = false;
  int? fees;
  String? duration;
  String? error;

  Future<void> _inquire() async {
    setState(() {
      loading = true;
      error = null;
    });

    try {
      // إجراء الاستعلامات بشكل متوازي
      final results = await Future.wait([
        http.get(
            Uri.parse(
                'http://100.84.28.28:8000/current-fees?car_plat=${widget.carNumber}'),
            headers: {'accept': 'application/json'}),
        http.get(
            Uri.parse(
                'http://100.84.28.28:8000/current-duration?car_plat=${widget.carNumber}'),
            headers: {'accept': 'application/json'}),
      ]);

      final feesResp = results[0];
      final durationResp = results[1];

      if (feesResp.statusCode == 200 && durationResp.statusCode == 200) {
        final feesData = jsonDecode(feesResp.body);
        final durationData = jsonDecode(durationResp.body);

        if (mounted) {
          setState(() {
            fees = feesData['fees'];
            duration = durationData['time_inside'];
          });
        }
      } else {
        if (mounted) {
          setState(() {
            error =
                'فشل الاستعلام: ${feesResp.statusCode} - ${durationResp.statusCode}';
          });
        }
      }
    } catch (e) {
      log('Error in _inquire: $e');
      if (mounted) {
        setState(() {
          error = 'حدث خطأ أثناء الاتصال بالخادم: ${e.toString()}';
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          loading = false;
        });
      }
    }
  }

  void _showPaymentSuccessDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: const Text('تم الدفع بنجاح!',
            textAlign: TextAlign.center,
            style: TextStyle(fontWeight: FontWeight.bold)),
        content: const Text('تمت عملية الدفع بنجاح. يمكنك الآن بدء حجز جديد.'),
        actionsAlignment: MainAxisAlignment.center,
        actions: [
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF10B981),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
            onPressed: () {
              Navigator.of(context).pushAndRemoveUntil(
                MaterialPageRoute(builder: (_) => AuthScreen()),
                (route) => false,
              );
            },
            child: const Text('بداية حجز جديد',
                style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  Future<void> _confirmPayment() async {
    setState(() {
      loading = true;
      error = null;
    });
    try {
      final response = await http.post(
        Uri.parse('http://100.84.28.28:8000/confirm-payment'),
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: {'car_plat': widget.carNumber},
      );
      final body = response.body;
      bool paymentSuccess = false;
      if (response.statusCode == 200) {
        paymentSuccess = true;
      } else if (response.statusCode == 404 &&
          body.contains('No unpaid booking found')) {
        paymentSuccess = true;
      } else {
        try {
          final decoded = jsonDecode(body);
          if (decoded is Map &&
              decoded['detail'] == 'Payment confirmed successfully.') {
            paymentSuccess = true;
          }
        } catch (_) {}
      }
      if (paymentSuccess) {
        _showPaymentSuccessDialog();
      } else {
        showDialog(
          context: context,
          builder: (ctx) => Directionality(
            textDirection: TextDirection.rtl,
            child: AlertDialog(
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20)),
              title: const Text('فشل الدفع',
                  style: TextStyle(
                      color: Colors.red, fontWeight: FontWeight.bold)),
              content: const Text(
                  'حدث خطأ أثناء عملية الدفع. الرجاء المحاولة مرة أخرى أو التأكد من الاتصال بالخادم.'),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(ctx).pop(),
                  child: const Text('حسناً'),
                ),
              ],
            ),
          ),
        );
      }
    } catch (e) {
      setState(() {
        error = 'فشل الاتصال بالخادم!';
      });
    } finally {
      if (mounted)
        setState(() {
          loading = false;
        });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Container(
        constraints: const BoxConstraints(maxWidth: 400),
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // أيقونة النجاح
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.green.withOpacity(0.1),
                borderRadius: BorderRadius.circular(50),
              ),
              child:
                  const Icon(Icons.check_circle, color: Colors.green, size: 48),
            ),
            const SizedBox(height: 16),

            // عنوان النجاح
            Directionality(
              textDirection: TextDirection.rtl,
              child: const Text(
                'تم الحجز بنجاح!',
                style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: Colors.green),
                textAlign: TextAlign.center,
              ),
            ),
            const SizedBox(height: 12),

            // تفاصيل الحجز
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.grey.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Directionality(
                textDirection: TextDirection.rtl,
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text('المنطقة:',
                            style: TextStyle(fontWeight: FontWeight.bold)),
                        Text(widget.zone, style: const TextStyle(fontSize: 16)),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text('رقم السيارة:',
                            style: TextStyle(fontWeight: FontWeight.bold)),
                        Text(widget.carNumber,
                            style: const TextStyle(fontSize: 16)),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 20),

            // معلومات الرسوم والمدة
            if (fees != null && duration != null) ...[
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.blue.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Directionality(
                  textDirection: TextDirection.rtl,
                  child: Column(
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text('الرسوم الحالية:',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          Text('$fees جنيه',
                              style: const TextStyle(
                                  fontSize: 16, color: Colors.blue)),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text('المدة:',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          Text(duration!, style: const TextStyle(fontSize: 16)),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ] else if (loading) ...[
              Container(
                padding: const EdgeInsets.all(16),
                child: const Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 8),
                    Text('جاري الاستعلام عن الرسوم والمدة...'),
                  ],
                ),
              ),
            ] else if (error != null) ...[
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.red.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  children: [
                    const Icon(Icons.error, color: Colors.red),
                    const SizedBox(height: 8),
                    Text(
                      error!,
                      style: const TextStyle(color: Colors.red),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ],

            const SizedBox(height: 12),

            // الأزرار
            Directionality(
              textDirection: TextDirection.rtl,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  ElevatedButton.icon(
                    onPressed: loading ? null : _inquire,
                    icon: const Icon(Icons.info_outline),
                    label: const Text('استعلام عن الرسوم'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 10),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),
                  const SizedBox(height: 5),
                  ElevatedButton.icon(
                    onPressed: loading ? null : _confirmPayment,
                    icon: const Icon(Icons.payment),
                    label: const Text('دفع الرسوم'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF10B981),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 10),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),
                  const SizedBox(height: 5),
                  OutlinedButton(
                    onPressed: () => Navigator.of(context).pop(),
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(
                          vertical: 10, horizontal: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                    child: const Text('إغلاق'),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _EnhancedBookingDialog extends StatefulWidget {
  const _EnhancedBookingDialog();

  @override
  State<_EnhancedBookingDialog> createState() => _EnhancedBookingDialogState();
}

class _EnhancedBookingDialogState extends State<_EnhancedBookingDialog>
    with TickerProviderStateMixin {
  double cost = 0.0;
  late Timer timer;
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    )..repeat(reverse: true);

    _pulseAnimation = Tween<double>(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    timer = Timer.periodic(const Duration(seconds: 1), (t) {
      setState(() {
        cost += 0.35;
      });
    });
  }

  @override
  void dispose() {
    timer.cancel();
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      backgroundColor: Colors.white,
      child: Padding(
        padding: const EdgeInsets.all(32.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
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
                        colors: [Color(0xFFFA709A), Color(0xFFFEE140)],
                      ),
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: [
                        BoxShadow(
                          color: const Color(0xFFFA709A).withOpacity(0.4),
                          blurRadius: 20,
                          offset: const Offset(0, 10),
                        ),
                      ],
                    ),
                    child: const Icon(
                      Icons.timer,
                      color: Colors.white,
                      size: 32,
                    ),
                  ),
                );
              },
            ),
            const SizedBox(height: 24),
            const Text(
              'الحجز قيد التنفيذ',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Color(0xFF1F2937),
              ),
            ),
            const SizedBox(height: 20),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 20),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFFFA709A), Color(0xFFFEE140)],
                ),
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFFFA709A).withOpacity(0.3),
                    blurRadius: 16,
                    offset: const Offset(0, 8),
                  ),
                ],
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.attach_money, color: Colors.white, size: 28),
                  const SizedBox(width: 12),
                  Text(
                    '${cost.toStringAsFixed(2)} ريال',
                    style: const TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.w800,
                      color: Colors.white,
                      letterSpacing: -1,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 32),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFFEF4444),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                      elevation: 8,
                      shadowColor: const Color(0xFFEF4444).withOpacity(0.4),
                    ),
                    onPressed: () {
                      HapticFeedback.mediumImpact();
                      timer.cancel();
                      Navigator.pop(context);
                    },
                    icon: const Icon(Icons.stop_circle, size: 20),
                    label: const Text(
                      'إنهاء الحجز',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// صفحة التاريخ
class _UserHistoryScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Text(
        'تاريخ معاملات المستخدم',
        style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
      ),
    );
  }
}

// صفحة الملف الشخصي
class _UserProfileScreen extends StatelessWidget {
  final String carNumber;
  const _UserProfileScreen({super.key, required this.carNumber});
  @override
  Widget build(BuildContext context) {
    final user = {
      'name': 'أحمد محمد',
      'email': 'ahmed@email.com',
      'image': 'assets/profile.png',
    };
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircleAvatar(
              radius: 50,
              backgroundImage: AssetImage(user['image']!),
            ),
            const SizedBox(height: 20),
            Text(
              user['name']!,
              style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text(
              user['email']!,
              style: const TextStyle(fontSize: 16, color: Colors.grey),
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.directions_car, color: Colors.deepPurple),
                const SizedBox(width: 6),
                Text(
                  carNumber,
                  style: const TextStyle(
                      fontSize: 16, fontWeight: FontWeight.w500),
                ),
              ],
            ),
            const SizedBox(height: 24),
            ListTile(
              leading: const Icon(Icons.settings),
              title: const Text('الإعدادات'),
              onTap: () {},
            ),
            ListTile(
              leading: const Icon(Icons.history),
              title: const Text('سجل المعاملات'),
              onTap: () {},
            ),
            ListTile(
              leading: const Icon(Icons.logout, color: Colors.red),
              title: const Text('تسجيل الخروج',
                  style: TextStyle(color: Colors.red)),
              onTap: () {},
            ),
          ],
        ),
      ),
    );
  }
}

// صفحة الإعدادات
class _SettingsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Text(
        'الإعدادات',
        style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
      ),
    );
  }
}
