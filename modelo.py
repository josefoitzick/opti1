import pulp as pl
import json
import os

def solve_instance(data, file_out, instance_name):
    """
    Resuelve el problema de asignación de doctores para cubrir la demanda de pacientes
    en distintas actividades, días y turnos, y escribe resultados en file_out.
    """

    # Extraemos la información principal
    dias = data["dias"]                     # Lista de días
    turnos = data["turnos"]                 # Lista de turnos
    categorias = data["categorias"]         # Lista de categorías
    actividades = data["actividades"]       # Lista de actividades

    demanda = data["demanda"]               # Diccionario anidado: demanda[dia][turno][categoria]
    tiempo_atencion = data["tiempo_atencion"]  # tiempo_atencion[categoria]
    min_personal = data["min_personal"]        # min_personal[actividad]
    duracion_turno = data["duracion_turno"]    # Ej: 720 minutos

    # Creamos el modelo
    model = pl.LpProblem("Minimizar_Doctores", pl.LpMinimize)

    # Variables de decisión: y[a, d, t] = número de doctores para la actividad 'a'
    # en el día 'd' y el turno 't'
    y = {}
    for a in actividades:
        for d in dias:
            for t in turnos:
                var_name = f"y_{a}_{d}_{t}"
                y[(a, d, t)] = pl.LpVariable(var_name, lowBound=0, cat=pl.LpInteger)

    # Función objetivo: Minimizar la suma total de doctores
    model += pl.lpSum(y[(a, d, t)] for a in actividades for d in dias for t in turnos), "Min_Doctores_Totales"

    # Restricciones

    # 1) Cobertura de la demanda: y[a,d,t] * duracion_turno >= sum(demanda * tiempo_atencion)
    for a in actividades:
        for d in dias:
            for t in turnos:
                workload = sum(
                    demanda[d][str(t)][c] * tiempo_atencion[c]
                    for c in categorias
                )
                model += (
                    y[(a, d, t)] * duracion_turno >= workload,
                    f"Cobertura_{a}_{d}_{t}"
                )

    # 2) Mínimo de personal por actividad
    for a in actividades:
        for d in dias:
            for t in turnos:
                model += (
                    y[(a, d, t)] >= min_personal[a],
                    f"Min_Personal_{a}_{d}_{t}"
                )

    # Resolver el problema
    solver = pl.PULP_CBC_CMD(msg=0)
    status = model.solve(solver)

    with open(file_out, "a", encoding="utf-8") as f:
        f.write(f"\n=== Instancia: {instance_name} ===\n")
        f.write(f"Estado de la solución: {pl.LpStatus[model.status]}\n")
        if pl.LpStatus[model.status] == "Optimal":
            valor_objetivo = pl.value(model.objective)
            f.write(f"Valor óptimo (Total de doctores): {valor_objetivo:.2f}\n\n")
            for a in actividades:
                for d in dias:
                    for t in turnos:
                        f.write(
                            f"  {a} - {d} - Turno {t}: "
                            f"{pl.value(y[(a, d, t)])} doctores\n"
                        )
        else:
            f.write("  No se encontró solución óptima o el problema es infactible.\n")

def main():
    carpeta_instancias = "instancias_json"
    archivo_resultados = "resultados.txt"

    # Borramos archivo de resultados si existe, para iniciar "limpio"
    if os.path.exists(archivo_resultados):
        os.remove(archivo_resultados)

    # Iteramos sobre todos los archivos .json de la carpeta
    for filename in sorted(os.listdir(carpeta_instancias)):
        if filename.endswith(".json"):
            ruta = os.path.join(carpeta_instancias, filename)
            try:
                with open(ruta, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"Resolviendo instancia: {filename}")
                solve_instance(data, archivo_resultados, instance_name=filename)
            except Exception as e:
                print(f"Error en instancia {filename}: {str(e)}")
                with open(archivo_resultados, "a", encoding="utf-8") as f:
                    f.write(f"\n=== Instancia: {filename} ===\n")
                    f.write(f"Error al procesar la instancia: {str(e)}\n")

if __name__ == "__main__":
    main()
