import {
  createRouter,
  createWebHistory,
  RouteRecordRaw,
} from "vue-router";

const routes: RouteRecordRaw[] = [
  {
    path: "/",
    component: {
      template: "",
    },
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
