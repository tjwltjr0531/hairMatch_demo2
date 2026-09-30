/*
  config.js — a.html 설정. a.html 을 새 버전으로 교체해도 이 파일은 그대로 두면 된다.
  바꾼 뒤에는 브라우저에서 Ctrl+F5.

  ★ 여기에는 브라우저에 공개돼도 되는 값만 적는다.
    - 카카오 JavaScript 키: 등록한 도메인에서만 동작하므로 OK
    - 카카오 REST 키, Gemini 키: 절대 금지 (서버 환경변수에만)
*/
window.HM_CONFIG = {
  MAP_KEY    : '5778ba061e623cdace952d5862626a9b',      // 카카오 JavaScript 키
  // USE_MOCK   : false, // 비워 두면 서버로 열 때 자동으로 실제 합성
  // SALON_MOCK : false,
  GEN_MOCK   : true,    // Gemini 결제 연결 후 false
};
