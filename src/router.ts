import {
  createRouter,
  createWebHistory,
  RouteRecordRaw,
} from "vue-router";
import PredictionData from './components/PredictionData.vue';

const routes: RouteRecordRaw[] = [
  {
    path: "/",
    component: {
      template: "",
    },
  },
  {
    path: "/predict",
    component: PredictionData,
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
