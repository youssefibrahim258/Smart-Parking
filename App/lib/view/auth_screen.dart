import 'dart:developer';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../view model/auth_view_model.dart';
import 'home_screen.dart';
import 'qr_scan_screen.dart';
import '../../global_timer_provider.dart';

class AuthScreen extends StatefulWidget {
  const AuthScreen({super.key});

  @override
  State<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends State<AuthScreen> with TickerProviderStateMixin {
  // متغير لرسالة الخطأ أو النجاح
  String? _serverMessage;
  bool _isSubmitting = false;

  // دالة التسجيل
  Future<void> _registerUser() async {
    setState(() {
      _isSubmitting = true;
      _serverMessage = null;
    });
    try {
      log("start");
      var headers = {'Content-Type': 'application/x-www-form-urlencoded'};
      var request =
          http.Request('POST', Uri.parse('http://100.84.28.28:8000/register'));
      request.bodyFields = {'email': email, 'password': password};
      request.headers.addAll(headers);
      var response = await request.send();
      var body = await response.stream.bytesToString();
      log(body);
      if (response.statusCode == 200) {
        final data = jsonDecode(body);
        setState(() {
          _serverMessage = data['message'] ?? 'تم إنشاء الحساب بنجاح';
        });
        Provider.of<GlobalTimerProvider>(context, listen: false).start();
        WidgetsBinding.instance.addPostFrameCallback((_) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (_) => QrScanScreen(email: email)),
          );
        });
      } else {
        final data = jsonDecode(body);
        setState(() {
          _serverMessage = data['detail'] ?? 'حدث خطأ أثناء التسجيل';
        });
      }
    } catch (e) {
      setState(() {
        _serverMessage = 'فشل الاتصال بالخادم';
      });
    } finally {
      setState(() {
        _isSubmitting = false;
      });
    }
  }

  // دالة تسجيل الدخول
  Future<void> _loginUser() async {
    setState(() {
      _isSubmitting = true;
      _serverMessage = null;
    });
    try {
      var headers = {'Content-Type': 'application/x-www-form-urlencoded'};
      var request =
          http.Request('POST', Uri.parse('http://100.84.28.28:8000/login'));
      request.bodyFields = {'email': email, 'password': password};
      request.headers.addAll(headers);
      var response = await request.send();
      var body = await response.stream.bytesToString();
      if (response.statusCode == 200) {
        final data = jsonDecode(body);
        setState(() {
          _serverMessage = 'تم تسجيل الدخول بنجاح';
        });
        Provider.of<GlobalTimerProvider>(context, listen: false).start();
        WidgetsBinding.instance.addPostFrameCallback((_) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (_) => QrScanScreen(email: email)),
          );
        });
      } else {
        final data = jsonDecode(body);
        setState(() {
          _serverMessage = data['detail'] ?? 'حدث خطأ أثناء تسجيل الدخول';
        });
      }
    } catch (e) {
      setState(() {
        _serverMessage = 'فشل الاتصال بالخادم';
      });
    } finally {
      setState(() {
        _isSubmitting = false;
      });
    }
  }

  bool isLogin = true;
  final _formKey = GlobalKey<FormState>();
  String email = '';
  String password = '';
  String carNumber = '';
  bool _isPasswordVisible = false;
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    );
    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeInOut,
    ));
    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 0.3),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeOutCubic,
    ));
    _animationController.forward();
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final auth = Provider.of<AuthViewModel>(context);
    final size = MediaQuery.of(context).size;

    return Directionality(
      textDirection: TextDirection.rtl,
      child: Scaffold(
        body: Container(
          height: size.height,
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Color(0xFF667eea),
                Color(0xFF764ba2),
                Color(0xFFf093fb),
              ],
            ),
          ),
          child: SafeArea(
            child: Center(
              child: SingleChildScrollView(
                child: Padding(
                  padding: const EdgeInsets.all(24.0),
                  child: FadeTransition(
                    opacity: _fadeAnimation,
                    child: SlideTransition(
                      position: _slideAnimation,
                      child: Container(
                        width: size.width > 400 ? 400 : double.infinity,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.95),
                          borderRadius: BorderRadius.circular(32),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.1),
                              blurRadius: 40,
                              offset: const Offset(0, 8),
                            ),
                          ],
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(32),
                          child: Container(
                            decoration: BoxDecoration(
                              border: Border.all(
                                color: Colors.white.withOpacity(0.2),
                                width: 1,
                              ),
                            ),
                            child: Padding(
                              padding: const EdgeInsets.all(40.0),
                              child: Form(
                                key: _formKey,
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    // شعار أو أيقونة
                                    Container(
                                      width: 80,
                                      height: 80,
                                      decoration: BoxDecoration(
                                        gradient: const LinearGradient(
                                          colors: [
                                            Color(0xFF667eea),
                                            Color(0xFF764ba2)
                                          ],
                                        ),
                                        borderRadius: BorderRadius.circular(40),
                                        boxShadow: [
                                          BoxShadow(
                                            color: const Color(0xFF667eea)
                                                .withOpacity(0.3),
                                            blurRadius: 20,
                                            offset: const Offset(0, 4),
                                          ),
                                        ],
                                      ),
                                      child: const Icon(
                                        Icons.person_rounded,
                                        color: Colors.white,
                                        size: 40,
                                      ),
                                    ),
                                    const SizedBox(height: 32),

                                    // العنوان
                                    Text(
                                      isLogin
                                          ? "مرحباً بك مرة أخرى!"
                                          : "إنشاء حساب جديد",
                                      style: const TextStyle(
                                        fontSize: 28,
                                        fontWeight: FontWeight.bold,
                                        color: Color(0xFF2D3748),
                                        height: 1.2,
                                      ),
                                      textAlign: TextAlign.center,
                                    ),
                                    const SizedBox(height: 8),
                                    Text(
                                      isLogin
                                          ? "قم بتسجيل الدخول للمتابعة"
                                          : "أنشئ حساباً جديداً للبدء",
                                      style: TextStyle(
                                        fontSize: 16,
                                        color: Colors.grey[600],
                                        height: 1.4,
                                      ),
                                      textAlign: TextAlign.center,
                                    ),
                                    const SizedBox(height: 40),

                                    // حقل البريد الإلكتروني
                                    _buildModernTextField(
                                      label: "البريد الإلكتروني",
                                      icon: Icons.email_rounded,
                                      keyboardType: TextInputType.emailAddress,
                                      validator: (val) => val!.contains('@')
                                          ? null
                                          : "أدخل بريد إلكتروني صحيح",
                                      onSaved: (val) => email = val!,
                                    ),
                                    const SizedBox(height: 24),

                                    // حقل كلمة المرور
                                    _buildModernTextField(
                                      label: "كلمة المرور",
                                      icon: Icons.lock_rounded,
                                      obscureText: !_isPasswordVisible,
                                      suffixIcon: IconButton(
                                        icon: Icon(
                                          _isPasswordVisible
                                              ? Icons.visibility_off_rounded
                                              : Icons.visibility_rounded,
                                          color: const Color(0xFF667eea),
                                        ),
                                        onPressed: () {
                                          setState(() {
                                            _isPasswordVisible =
                                                !_isPasswordVisible;
                                          });
                                        },
                                      ),
                                      validator: (val) => val!.length >= 6
                                          ? null
                                          : "كلمة المرور قصيرة جداً",
                                      onSaved: (val) => password = val!,
                                    ),
                                    const SizedBox(height: 32),

                                    // رسالة الخطأ
                                    if (auth.error != null)
                                      Container(
                                        padding: const EdgeInsets.all(12),
                                        margin:
                                            const EdgeInsets.only(bottom: 16),
                                        decoration: BoxDecoration(
                                          color: Colors.red.shade50,
                                          borderRadius:
                                              BorderRadius.circular(12),
                                          border: Border.all(
                                              color: Colors.red.shade200),
                                        ),
                                        child: Row(
                                          children: [
                                            Icon(Icons.error_outline,
                                                color: Colors.red.shade600,
                                                size: 20),
                                            const SizedBox(width: 8),
                                            Expanded(
                                              child: Text(
                                                auth.error!,
                                                style: TextStyle(
                                                  color: Colors.red.shade700,
                                                  fontSize: 14,
                                                ),
                                              ),
                                            ),
                                          ],
                                        ),
                                      ),

                                    // زر تسجيل الدخول/إنشاء الحساب
                                    SizedBox(
                                      width: double.infinity,
                                      height: 56,
                                      child: ElevatedButton(
                                        style: ElevatedButton.styleFrom(
                                          backgroundColor: Colors.transparent,
                                          shadowColor: Colors.transparent,
                                          shape: RoundedRectangleBorder(
                                            borderRadius:
                                                BorderRadius.circular(16),
                                          ),
                                        ).copyWith(
                                          backgroundColor:
                                              WidgetStateProperty.all(
                                                  Colors.transparent),
                                        ),
                                        onPressed: _isSubmitting
                                            ? null
                                            : () async {
                                                if (_formKey.currentState!
                                                    .validate()) {
                                                  _formKey.currentState!.save();
                                                  if (isLogin) {
                                                    await _loginUser();
                                                  } else {
                                                    await _registerUser();
                                                  }
                                                }
                                              },
                                        child: Container(
                                          width: double.infinity,
                                          height: 56,
                                          decoration: BoxDecoration(
                                            gradient: const LinearGradient(
                                              colors: [
                                                Color(0xFF667eea),
                                                Color(0xFF764ba2)
                                              ],
                                            ),
                                            borderRadius:
                                                BorderRadius.circular(16),
                                            boxShadow: [
                                              BoxShadow(
                                                color: const Color(0xFF667eea)
                                                    .withOpacity(0.3),
                                                blurRadius: 20,
                                                offset: const Offset(0, 4),
                                              ),
                                            ],
                                          ),
                                          child: Center(
                                            child: _isSubmitting
                                                ? const SizedBox(
                                                    width: 24,
                                                    height: 24,
                                                    child:
                                                        CircularProgressIndicator(
                                                      color: Colors.white,
                                                      strokeWidth: 2,
                                                    ),
                                                  )
                                                : Text(
                                                    isLogin
                                                        ? "تسجيل الدخول"
                                                        : "إنشاء حساب",
                                                    style: const TextStyle(
                                                      color: Colors.white,
                                                      fontSize: 18,
                                                      fontWeight:
                                                          FontWeight.w600,
                                                    ),
                                                  ),
                                          ),
                                        ),
                                      ),
                                    ),
                                    // رسالة الخادم (نجاح أو خطأ)
                                    if (_serverMessage != null)
                                      Padding(
                                        padding: const EdgeInsets.symmetric(
                                            vertical: 12),
                                        child: Text(
                                          _serverMessage!,
                                          style: TextStyle(
                                            color: _serverMessage!
                                                        .contains('نجاح') ||
                                                    _serverMessage!
                                                        .contains('تم')
                                                ? Colors.green
                                                : Colors.red,
                                            fontWeight: FontWeight.bold,
                                            fontSize: 16,
                                          ),
                                          textAlign: TextAlign.center,
                                        ),
                                      ),
                                    const SizedBox(height: 24),

                                    // زر التبديل بين تسجيل الدخول وإنشاء حساب
                                    TextButton(
                                      onPressed: auth.isLoading
                                          ? null
                                          : () {
                                              setState(() {
                                                isLogin = !isLogin;
                                              });
                                            },
                                      style: TextButton.styleFrom(
                                        padding: const EdgeInsets.symmetric(
                                            horizontal: 16, vertical: 8),
                                      ),
                                      child: Text(
                                        isLogin
                                            ? "ليس لديك حساب؟ أنشئ حساباً جديداً"
                                            : "لديك حساب بالفعل؟ سجل الدخول",
                                        style: const TextStyle(
                                          color: Color(0xFF667eea),
                                          fontSize: 16,
                                          fontWeight: FontWeight.w500,
                                        ),
                                        textAlign: TextAlign.center,
                                      ),
                                    ),

                                    const SizedBox(height: 16),

                                    // خط فاصل
                                    Row(
                                      children: [
                                        Expanded(
                                            child: Divider(
                                                color: Colors.grey[300])),
                                        Padding(
                                          padding: const EdgeInsets.symmetric(
                                              horizontal: 16),
                                          child: Text(
                                            "أو",
                                            style: TextStyle(
                                              color: Colors.grey[600],
                                              fontSize: 14,
                                            ),
                                          ),
                                        ),
                                        Expanded(
                                            child: Divider(
                                                color: Colors.grey[300])),
                                      ],
                                    ),

                                    const SizedBox(height: 24),

                                    // أزرار تسجيل الدخول الاجتماعي
                                    Row(
                                      mainAxisAlignment:
                                          MainAxisAlignment.spaceEvenly,
                                      children: [
                                        _buildSocialButton(
                                          icon: Icons.g_mobiledata,
                                          color: const Color(0xFF4285f4),
                                          onPressed: () {},
                                        ),
                                        _buildSocialButton(
                                          icon: Icons.facebook,
                                          color: const Color(0xFF1877f2),
                                          onPressed: () {},
                                        ),
                                        _buildSocialButton(
                                          icon: Icons.apple,
                                          color: Colors.black,
                                          onPressed: () {},
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
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildModernTextField({
    required String label,
    required IconData icon,
    TextInputType? keyboardType,
    bool obscureText = false,
    Widget? suffixIcon,
    String? Function(String?)? validator,
    void Function(String?)? onSaved,
  }) {
    return TextFormField(
      keyboardType: keyboardType,
      obscureText: obscureText,
      validator: validator,
      onSaved: onSaved,
      style: const TextStyle(
        fontSize: 16,
        color: Color(0xFF2D3748),
      ),
      decoration: InputDecoration(
        labelText: label,
        labelStyle: TextStyle(
          color: Colors.grey[600],
          fontSize: 16,
          fontWeight: FontWeight.w500,
        ),
        prefixIcon: Container(
          margin: const EdgeInsets.all(12),
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              colors: [Color(0xFF667eea), Color(0xFF764ba2)],
            ),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(
            icon,
            color: Colors.white,
            size: 20,
          ),
        ),
        suffixIcon: suffixIcon,
        filled: true,
        fillColor: Colors.grey[50],
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: Colors.grey[300]!),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: Colors.grey[300]!),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: Color(0xFF667eea), width: 2),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: Colors.red, width: 2),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: Colors.red, width: 2),
        ),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 20),
      ),
    );
  }

  Widget _buildSocialButton({
    required IconData icon,
    required Color color,
    required VoidCallback onPressed,
  }) {
    return Container(
      width: 56,
      height: 56,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.grey[300]!),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: onPressed,
          child: Center(
            child: Icon(
              icon,
              color: color,
              size: 24,
            ),
          ),
        ),
      ),
    );
  }
}
