import pygame

def createJOI(density, condition, vehicle_counter, green_light_duration):
    pygame.init()
    width, height = 512, 512
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    
    counter_15 = green_light_duration
    counter_14 = green_light_duration
    counter_13 = 13
    counter_12 = 12
    
    red_light_radius = 20
    red_light_position = (256, 256)  # Koordinat pusat lampu merah
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((255, 255, 255))
        
        pygame.draw.line(screen, (0, 0, 0), (0, 190), (170, 190), 2)
        pygame.draw.line(screen, (0, 0, 0), (0, 190), (0, 340), 2)
        pygame.draw.line(screen, (0, 0, 0), (0, 340), (170, 340), 2)
        pygame.draw.line(screen, (0, 0, 0), (170, 0), (170, 190), 2)
        pygame.draw.line(screen, (0, 0, 0), (170, 340), (170, 512), 2)
        pygame.draw.line(screen, (0, 0, 0), (320, 0), (320, 190), 2)
        pygame.draw.line(screen, (0, 0, 0), (320, 190), (512, 190), 2)
        pygame.draw.line(screen, (0, 0, 0), (320, 340), (512, 340), 2)
        pygame.draw.line(screen, (0, 0, 0), (320, 340), (320, 512), 2)
        
        font = pygame.font.Font(None, 20)
        text_density = font.render(f'Kepadatan: {density}', True, (0, 0, 0))
        screen.blit(text_density, (10, 20))
        text_condition = font.render(f'Kondisi: {condition}', True, (0, 0, 0))
        screen.blit(text_condition, (10, 60))
        text_vehicle_counter = font.render(f'Kendaraan: {vehicle_counter}', True, (0, 0, 0))
        screen.blit(text_vehicle_counter, (330, 60))
        text_green_light_duration = font.render(f'Durasi Lampu Hijau: {green_light_duration}', True, (0, 0, 0))
        screen.blit(text_green_light_duration, (330, 20))
        
        pygame.draw.rect(screen, (0, 0, 0), (130, 150, 40, 40))
        text_counter_15 = font.render(str(counter_15), True, (255, 255, 255))
        screen.blit(text_counter_15, (130, 180))
        
        pygame.draw.rect(screen, (0, 0, 0), (130, 340, 40, 40))
        text_counter_14 = font.render(str(counter_14), True, (255, 255, 255))
        screen.blit(text_counter_14, (130, 370))
        
        pygame.draw.rect(screen, (0, 0, 0), (320, 150, 40, 40))
        text_counter_13 = font.render(str(counter_13), True, (255, 255, 255))
        screen.blit(text_counter_13, (320, 180))
        
        pygame.draw.rect(screen, (0, 0, 0), (320, 340, 40, 40))
        text_counter_12 = font.render(str(counter_12), True, (255, 255, 255))
        screen.blit(text_counter_12, (320, 370))
        
        pygame.draw.circle(screen, (255, 0, 0), red_light_position, red_light_radius)
        
        pygame.display.flip()
        clock.tick(1)
        
        counter_15 -= 1
        counter_14 -= 1
        counter_13 -= 1
        counter_12 -= 1
        
        if counter_15 < 0:
            counter_15 = green_light_duration
        
        if counter_14 < 0:
            counter_14 = green_light_duration
        
        if counter_13 < 0:
            counter_13 = 13
        
        if counter_12 < 0:
            counter_12 = 12
    
    pygame.quit()

density = 10
vehicle_counter = 10
condition = 0
green_light_duration = 10

createJOI(density, condition, vehicle_counter, green_light_duration)
