// F9 자연어 대화 (Frontend-Route 14절) — 백엔드 F9_Chat_agent/chat/api.py 와 짝이다.
// 메시지는 서버가 만들어 저장하고 그대로 내려준다. `at` 은 'YYYY-MM-DDTHH:MM' 로컬.
// 쓰기(등록·수정·삭제)는 preview 메시지(카드)로만 오고, 확인(confirm)해야 서버가 실행한다.

import type { CategoryKey } from "./types";

export type ChatMsg =
  | { id: string; role: "user"; text: string; at: string }
  | {
      id: string;
      role: "assistant";
      text: string;
      at: string;
      /** 목록 카드 — when 은 서버가 만든 표시 문자열 (예 "오늘 23:59" · "11/6(금) 18:00") */
      list?: { title: string; when: string; sub: string; href?: string }[];
      /** 근거 화면 링크 (내부 경로만) */
      link?: { label: string; href: string };
    }
  | {
      /** == PendingAction id */
      id: string;
      role: "preview";
      at: string;
      kind: "create" | "update" | "delete";
      /** start/end: 종일 "YYYY-MM-DD"(end exclusive), 아니면 "YYYY-MM-DDTHH:MM" */
      title: string;
      start: string;
      end: string;
      allDay: boolean;
      category: "personal" | "study" | "team" | "etc";
      categoryLabel: string;
      /** update/delete 대상(숫자 id) */
      targetId?: string;
      /** update: 바꾸기 전 값 */
      before?: { title: string; start: string; end: string; allDay: boolean; category: string };
      /** "겹치는 일정 1건 — 팀 회의 20:00~21:00" */
      conflict?: string;
      /** "지난 날짜입니다" 등 (확인하면 그대로 등록) */
      warning?: string;
      state: "pending" | "confirmed" | "canceled" | "expired";
      expiresAt: string;
    }
  /** 실행 기록: "e클래스 수집을 시작했습니다", 오류 안내 */
  | { id: string; role: "system"; text: string; at: string };

export type ChatPreview = Extract<ChatMsg, { role: "preview" }>;
export type ChatListItem = NonNullable<Extract<ChatMsg, { role: "assistant" }>["list"]>[number];

/** GET /api/chat/status · /api/status 의 chat 칸 */
export interface ChatStatus {
  available: boolean;
  llmConfigured: boolean;
  error?: string;
  model?: string;
}

/** POST /api/chat — messages 는 이번 턴에 새로 생긴 것(사용자 메시지 echo 포함, 모두 저장됨) */
export interface ChatTurn {
  reply: string;
  messages: ChatMsg[];
}

/** POST /api/chat/actions/{id}/confirm — 카드에서 고친 값 (create/update 만) */
export interface PreviewEdits {
  title?: string;
  start?: string;
  end?: string;
  allDay?: boolean;
  category?: CategoryKey;
}
